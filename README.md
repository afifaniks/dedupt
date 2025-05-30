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
python src/main.py dedupt --data_path ${DATASET_ROOT}/netbeans_2016/netbeans_stacktraces.json --bucket_name netbeans --trim_len 0 --lang java --multi_stack --loss ranknet --encoder_path ${model_save_path}

Add --skip_training if you have already trained and want to evaluate an existing checkpoint.
```
At the end of the training, the training and evaluation summary will be shown and the best checkpoint will be saved to `artifacts/sim_model_*.pth`.

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
To run baselines like S3M or DeepCrash, just pass one of the method name:
```bash
python src/main.py {s3m, deepcrash} --data_path ${DATASET_ROOT}/netbeans_2016/netbeans_stacktraces.json --bucket_name netbeans --trim_len 0 --lang java --loss ranknet
```
Rest of the baselines can be ran, using the following this repository and README: https://github.com/irving-muller/TraceSim_EMSE

# Additional Results
### Table 1: Comparison of the LLM-based approach and ours on different datasets.

#### Table 1: Comparison of the LLM-based approach and ours on different datasets.

| Dataset   | Approach                   | Embedding Model         | MRR   | RR@1  | RR@5  |
|-----------|----------------------------|--------------------------|-------|-------|-------|
| Ubuntu    | VectorDB            | text-embedding-3-small   | 0.118 | 0.000 | 0.103 |
|           | VectorDB + GPT-4o   | text-embedding-3-small   | 0.584 | 0.427 | 0.761 |
|           |Ours                        | bge-base-en              | 0.786 | 0.744 | 0.838 |
| Netbeans  | VectorDB            | text-embedding-3-small   | 0.121 | 0.000 | 0.101 |
|           | VectorDB + GPT-4o   | text-embedding-3-small   | 0.586 | 0.422 | 0.825 |
|           | Ours                        | bge-base-en              | 0.771 | 0.681 | 0.878 |

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