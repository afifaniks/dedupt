# DedupT
A transformer based method to stack trace-based bug deduplication

# Data
We utilized data used in this study: https://github.com/irving-muller/TraceSim_EMSE

Dataset Link: https://zenodo.org/record/5746044#.YabKILtyZH4

Before proceeding to the next steps, please download and extract the datasets in your machine.

# Setup

To run this project, we recommend using Python 3.8+ with a seperate conda environment.

Install requirements:
```bash
pip install -r requirements.txt
```
# Embedding Model Fine Tuning

The following example is for the Netbeans dataset. Available datsets/buckets: 

    - netbeans
    - eclipse
    - ubuntu
    - gnome

To fine tune an embedding model for a specific bug dataset:
```bash
Navigate to project root.

Create a new directory (`artifacts`) where we can save some artifacts and record the path for future reference.

mkdir artifacts

python src/adapt.py --bucket netbeans --data_root ${DATASET_ROOT} --artifact_dir ${PROJECT_ROOT/artifacts} --plm "BAAI/bge-base-en" --trim_len 0 --max_frames 10
```

Once the embedding model is fine-tuned, it should produce a `JSON` file in the `artifacts` directory summarizing the fine-tuning process. 

Open the file and there should be a field `model_save_path` having the file path for the fine-tuned embedding model for the future steps.

# Classifier Training
Execute the following.

```bash
python src/main.py dedupt --data_path ${DATASET_ROOT}/netbeans_2016/netbeans_stacktraces.json --bucket_name netbeans --trim_len 0 --lang java --multi_stack --loss ranknet --encoder_path ${model_save_path} --result_path ${PREDICTION_FILE_PATH}.json

# Add --skip_training if you have already trained and want to evaluate an existing checkpoint.
# Use --result_path to save the prediction on test data after the model is trained
```
At the end of the training, the training and evaluation summary will be shown and the best checkpoint will be saved to `artifacts/sim_model_*.pth`.

The `--result_path` is necessary to save predictions in a .json file for further analysis like statistical test.

# OpenAI Embedding-based Pipeline
This pipeline will replace our encoder model with a proprietary OpenAI model and start training on top of it. If you want to run the evaluation using OpenAI embedding model `text-embedding-3-small`, use the following"
```bash
python src/main.py dedupt --data_path ${DATASET_ROOT}/netbeans_2016/netbeans_stacktraces.json --bucket_name netbeans --trim_len 0 --lang java --multi_stack --loss ranknet --encoder_path text-embedding-3-small
```

# LLM Pipeline
We also introduced a GPT-4o based pipeline to compare with our approach. To use this, please create a `.env` file in the project root and set the environment variable.
```bash
OPENAI_API_KEY={YOUR_KEY}
```
Then run:
```bash
python src/main_llm.py llm --data_path ${DATASET_ROOT}/netbeans_2016/netbeans_stacktraces.json --bucket_name netbeans --trim_len 0 --lang java --multi_stack --loss ranknet
```
Please note that, there is no additional training required for this pipeline. This will automatically create a vector database using `text-embedding-3-small` and then immediately start evaluating on the test data.

# Baselines

### To run baseline S3M, DeepCrash, just pass one of the method name:
```bash
python src/main.py {s3m, deepcrash} --data_path ${DATASET_ROOT}/netbeans_2016/netbeans_stacktraces.json --bucket_name netbeans --trim_len 0 --lang java --multi_stack --loss ranknet
```

### To run the other baselines like Tracesim, Rebucket, Needleman, etc., we use the pipeline developed by Rodrigues et al., 2022: https://github.com/irving-muller/TraceSim_EMSE

All the required dependencies should be installed following the official README.

## Dataset Preparation for TraceSim

To ensure all the baslines in this repository runs on the same train/val/test split as dedupT, we provide a helper script: [TraceSim_EMSE-main/generate_tracesim_data_split.py](TraceSim_EMSE-main/generate_tracesim_data_split.py)

For example, to generate similar splits as ours with 4200 train days, 140 validation days, and 700 test days on Netbeans dataset:
```bash
python TraceSim_EMSE-main/generate_tracesim_data_split.py --dataset ${DATASET_ROOT}/netbeans_2016/netbeans_stacktraces.json --out_dir ${DATA_OUTPUT_DIRECTORY} --train_days 3850 --warmup_days 350 --test_days 700 --val_days 140
# Train days = train_days (3850) + warmup_days (350) = 4200
```
This should result in 3 different splits/chunks that we can use for our training and evaluation of the baselines:

```
OUTPUT_DIRECTORY/
    --- test_chunk_0.txt
    --- training_chunk_0.txt
    --- validation_chunk_0.txt
```

Using these chunks, we can run all the baselines by following the instructions provided in this [README.](TraceSim_EMSE-main/README.md)

However, to run staistical significance test comparing our approach and any baseline from this pipeline, we need to save the predictions of both approaches.

For example, if we want to run Tracesim on Netbeans dataset and save predictions:

```bash
python TraceSim_EMSE-main/experiments/hyperparameter_opt.py \
  ${DATASET_ROOT}/netbeans_2016/netbeans_stacktraces.json \
  ${DATA_OUTPUT_DIRECTORY}/validation_chunk_0.txt \
  trace_sim \
  TraceSim_EMSE-main/space_script/trace_sim_space_netbeans.py \
  -test ${DATA_OUTPUT_DIRECTORY}/test_chunk_0.txt \
  -result_file ${PREDICTION_DIR}/tracesim_netbeans_results.sparse \
  -w 700 -max_evals 100 -nthreads 32 -filter_func threshold_trim -sparse
```
This will produce a .sparse file (`${PREDICTION_DIR}/tracesim_netbeans_results.sparse`) containing all the predictions on the test data which we can use for further analysis or statistical test. Note that this process will result in large .sparse files from every evaluation step and the final test step. For example, if there are 100 evaluation steps, there will be 101 .sparse files saved in the directory, starting with the provided result file name and the final test prediction file name should be `${PREDICTION_DIR}/tracesim_netbeans_results_100.sparse`.

This is the file we shall use for statistical test. Since these prediction files are very 

# Statistical Test

If all the required prediction files are saved, Wilcoxon statistical test can be run by using the following command:

```bash
python statistical_analysis/stat_test.py --dedupt_preds ${DEDUPT_PRED.json} --other_preds ${OTHER_PRED.sparse}
```
This should provide console output like this:

```bash
Wlicoxon stat: XXX, P-value: XXe-yy
    Result: *** Very significant (p < 0.01)
```

The script first reads the sparse data and pairs predictions from both methods and computes reciprocal rank and then performs wilcoxon signed rank test:

| bug_id | actual_duplicate_id | dedupt_preds                                   | other_method_preds                             | rr_dedupt | rr_other |
|--------|--------------------|-----------------------------------------------|------------------------------------------------|-----------|----------|
| 416153 | 407502             | [407502, 395935, 402412, 384263, 415789, ...] | [415946, 407502, 409876, 411851, 400538, ...]  | 1.0       | 0.500000 |
| 416155 | 407502             | [384263, 407502, 405921, 395935, 389685, ...] | [411851, 415946, 409876, 407502, 416153, ...]  | 0.5       | 0.250000 |
| 416273 | 410011             | [410011, 399690, 404388, 393054, 410509, ...] | [410011, 412518, 398324, 392561, 377593, ...]  | 1.0       | 1.000000 |
| 416382 | 390409             | [390409, 411208, 390072, 405821, 394344, ...] | [390409, 369663, 373897, 358774, 355722, ...]  | 1.0       | 1.000000 |
| 416383 | 390409             | [390409, 411208, 390072, 405821, 404162, ...] | [369245, 415925, 373298, 415512, 366785, ...]  | 1.0       | 0.000000 |


We provide a notebook demostrating how this is computed and how the results from different methods are paired: [statistical_analysis/stat_test.ipynb](statistical_analysis/stat_test.ipynb)

# Additional Results
### Table 1: Comparison of the LLM-based approach and ours on different datasets.

| Dataset   | Sample Size      | Approach                   | Embedding Model         | MRR   | RR@1  | RR@5  |
|-----------|------------------|----------------------------|--------------------------|-------|-------|-------|
| Ubuntu    | Full test set    | Vector Database            | text-embedding-small-3   | 0.118 | 0.000 | 0.103 |
|           |                  | Vector Database + GPT-4o   | text-embedding-small-3   | 0.584 | 0.427 | 0.761 |
|           |                  | Ours                        | bge-base-en              | 0.786 | 0.744 | 0.838 |
| Netbeans  | 100 test samples | Vector Database            | text-embedding-small-3   | 0.121 | 0.000 | 0.101 |
|           |                  | Vector Database + GPT-4o   | text-embedding-small-3   | 0.586 | 0.422 | 0.825 |
|           |                  | Ours                        | bge-base-en              | 0.752 | 0.651 | 0.863 |

### Table 2: Comparison of performance when CodeBERT is used as an embedding model.

| Dataset   | Approach  | MRR   | RR@1  | RR@5  | RR@10 |
|-----------|-----------|-------|-------|-------|-------|
| Netbeans  | CodeBERT  | 0.687 | 0.598 | 0.794 | 0.844 |
|           | Ours      | 0.771 | 0.681 | 0.878 | 0.918 |
| Eclipse   | CodeBERT  | 0.733 | 0.664 | 0.823 | 0.849 |
|           | Ours      | 0.791 | 0.720 | 0.880 | 0.902 |


### Table 3: Comparison of a SOTA OpenAI embedding model’s performance compared to our finetuned model.

| Dataset   | Embedding Model           | Output Dimension | MRR   | RR@1  | RR@5  |
|-----------|---------------------------|------------------|-------|-------|-------|
| Netbeans  | text-embedding-3-small    | 1536             | 0.696 | 0.603 | 0.800 |
|           | bge-base-en (finetuned)   | 512              | 0.771 | 0.681 | 0.878 |
| Ubuntu    | text-embedding-3-small    | 1536             | 0.707 | 0.637 | 0.797 |
|           | bge-base-en (finetuned)   | 512              | 0.786 | 0.744 | 0.838 |

### Table 4: Inference Efficiency Comparison for a Single Sentence with Batch Size 16

| Model                | Device      | GPU Memory Usage | Inference Time per Batch |
|----------------------|-------------|------------------|---------------------------|
| **bge-base**         | GPU         | 432 MB           | 0.01 s                    |
| **CodeBERT / BERT**  | GPU         | ~1.6 GB          | >0.03 s                   |
| **bge-base**         | CPU         | N/A              | ~0.28 s                   |
| **CodeBERT / BERT**  | CPU         | N/A              | ~3.3 s                    |

Devices used for this comparison\
CPU Used: Intel(R) Xeon(R) Gold 6148 CPU @ 2.40GHz\
GPU Used: NVIDIA V100 16GB
### Acknowledgmenet
This repository extended and used codes from the following repositories:
- https://github.com/akhvorov/S3M/tree/main
- https://github.com/irving-muller/TraceSim_EMSE