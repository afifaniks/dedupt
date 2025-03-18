# DedupT
A transformer based method to stack trace-based bug deduplication

# Data
We utilized data used in this study: https://github.com/irving-muller/TraceSim_EMSE

Dataset Link: https://zenodo.org/record/5746044#.YabKILtyZH4

Before proceeding to the next steps, please download and extract the datasets in your machine.

# Setup

To run this project we recommend using Python 3.10+ with a seperate conda environment.

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

### Acknowledgmenet
This repository extended and used codes from the following repositories:
- https://github.com/akhvorov/S3M/tree/main
- https://github.com/irving-muller/TraceSim_EMSE