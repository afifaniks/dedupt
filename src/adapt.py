# Disable wandb
import argparse
import json
import os
import random
import time

import numpy as np
import torch
from sentence_transformers import (SentenceTransformer,
                                   SentenceTransformerTrainer,
                                   SentenceTransformerTrainingArguments)
from sentence_transformers.evaluation import (EmbeddingSimilarityEvaluator,
                                              TripletEvaluator)
from sentence_transformers.losses import MultipleNegativesRankingLoss
from sentence_transformers.training_args import BatchSamplers
from tqdm import tqdm

from data.buckets.bucket_data import OtherBucketData
from data.buckets.issues_data import BucketDataset
from data.formatters import get_formatter
from data.triplet_selector import RandomTripletSelector
from datasets import Dataset
from preprocess.entry_coders import Stack2Seq, Stack2SeqMultiStack
from preprocess.seq_coder import SeqCoder, SeqCoderMulti
from preprocess.tokenizers import SimpleTokenizer

# Random seed
seed = 42
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
random.seed(seed)
np.random.seed(seed)

os.environ["WANDB_DISABLED"] = "true"

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")


parser = argparse.ArgumentParser()
parser.add_argument("--bucket", required=True, type=str, help="Bucket name")
parser.add_argument("--data_root", required=True, type=str, help="Dataset root directory")
parser.add_argument("--artifact_dir", required=True, type=str, help="Artifact root directory")
parser.add_argument("--plm", default="BAAI/bge-base-en", type=str, help="Sentence transformer model key")
parser.add_argument("--trim_len", default=0, type=int, help="Trim length for stack trace")
parser.add_argument("--max_frames", default=10, type=int, help="Maximum number of frames")
args = parser.parse_args()

print("Arguments:", args)

plm_key = args.plm
data_root = args.data_root
artifact_dir = args.artifact_dir
model = SentenceTransformer(plm_key, trust_remote_code=True)
print(f"Model sequence length: {model.max_seq_length}")

# model.max_seq_length = 2048

# Find model size in CUDA
print(f"Model size: {torch.cuda.memory_allocated() / 1024 ** 3:.2f} GB")

dataset_paths = {
    "eclipse": "eclipse_2018/eclipse_stacktraces.json",
    "netbeans": "netbeans_2016/netbeans_stacktraces.json",
    "gnome": "gnome_2011/gnome_stacktraces_filtered.json",
    "ubuntu": "campbell_dataset/campbell_stacktraces.json",
}

language_map = {
    "netbeans": "java",
    "eclipse": "java",
    "gnome": "cpp_pretrain",
    "ubuntu": "cpp",
}

bucket_name = args.bucket
dataset_json = os.path.join(data_root, dataset_paths[bucket_name]) 
language = language_map[bucket_name]
generate_dataset = True
batch_size = 16
eval_size = 300
max_num_frames = args.max_frames
stack_formatter = get_formatter(language, max_num_frames)
num_train_pairs = 5
num_test_pairs = 1
trim_length = args.trim_len
frame_freq = {}
test_only = False
warmup_days = 350
test_days = 700
val_days = 140
train_days = 3850
plm_key_norm = plm_key.replace("/", "_").strip().lower()
dataset_key = f"{bucket_name}_pretrain_dataset_{train_days}_traindays"
run_name = f"{plm_key_norm}_{bucket_name}_{max_num_frames}frames_{num_train_pairs}train_{trim_length}trim_{train_days}_traindays"
model_dir = os.path.join(artifact_dir, "trained_embedding_models")
datasets_dir = os.path.join(artifact_dir, "datasets")

os.makedirs(model_dir, exist_ok=True)
os.makedirs(datasets_dir, exist_ok=True)

config_to_write = {
    "dataset_key": dataset_key,
    "run_name": run_name,
    "train_days": train_days,
    "test_days": test_days,
    "val_days": val_days,
    "warmup_days": warmup_days,
    "plm_key": plm_key,
    "bucket_name": bucket_name,
    "artifact_dir": artifact_dir,
    "data_root": data_root,
    "dataset_json": dataset_json,
    "language": language,
    "generate_dataset": generate_dataset,
    "batch_size": batch_size,
    "eval_size": eval_size,
    "max_num_frames": max_num_frames,
    "num_train_pairs": num_train_pairs,
    "num_test_pairs": num_test_pairs,
    "frame_freq": frame_freq,
    "test_only": test_only,
    "trim_length": trim_length,
    "model_save_path": f"{model_dir}/{run_name}/final",
}

print("Configurations:\n", json.dumps(config_to_write, indent=2))

# Print all training parameters
print("Train pair:", num_train_pairs, "Test pair:", num_test_pairs)


def get_data_row(coder, event, similar_stack_id, dissimilar_stack_id):
    anchor_frames = coder(event.st_id, transformer=True)
    positive_frames = coder(similar_stack_id, transformer=True)
    negative_frames = coder(dissimilar_stack_id, transformer=True)

    return {
        "anchor": anchor_frames,
        "positive": positive_frames,
        "negative": negative_frames,
    }


def format_data_row(frames):
    return {
        "anchor": stack_formatter.format(frames["anchor"]),
        "positive": stack_formatter.format(frames["positive"]),
        "negative": stack_formatter.format(frames["negative"]),
    }


def generate_dataset_for_train_test(
    bucket_name,
    dataset_path,
    num_train_pairs,
    num_test_pairs,
    trim_length,
    test_only=False,
):
    triplet_selector_train = RandomTripletSelector(num_train_pairs)
    triplet_selector_eval = RandomTripletSelector(num_test_pairs)
    print("Load bucket data...")
    bucket_data = OtherBucketData(
        bucket_name,
        dataset_path,
        train_days,
        test_days,
        warmup_days,
        val_days,
        lang=language,
    )
    bucket_data.load()
    stack_loader = bucket_data.stack_loader()
    data_gen = BucketDataset(bucket_data)
    unsup_stacks = data_gen.train_stacks()

    data_gen.reset()

    stack2seq = None
    coder = None
    if bucket_name == "gnome":
        stack2seq = Stack2SeqMultiStack(cased=False, trim_len=trim_length, sep=".")
        coder = SeqCoderMulti(
            stack_loader, stack2seq, SimpleTokenizer(), min_freq=0, max_len=None
        )
    else:
        stack2seq = Stack2Seq(cased=False, trim_len=trim_length, sep=".")
        coder = SeqCoder(
            stack_loader, stack2seq, SimpleTokenizer(), min_freq=0, max_len=None
        )

    # coder.fit(unsup_stacks)

    train_data = []
    test_data = []

    print("Generate the dataset...")
    if not test_only:
        for i, event in tqdm(enumerate(data_gen.train()), desc="Generating train data"):
            similar_stack_ids, dissimilar_stack_ids = triplet_selector_train(event)
            for similar_stack_id, dissimilar_stack_id in zip(
                similar_stack_ids, dissimilar_stack_ids
            ):
                train_data.append(
                    get_data_row(
                        coder,
                        event,
                        similar_stack_id,
                        dissimilar_stack_id,
                    )
                )

    for i, event in tqdm(enumerate(data_gen.test()), desc="Generating test data"):
        similar_stack_ids, dissimilar_stack_ids = triplet_selector_eval(event)
        for similar_stack_id, dissimilar_stack_id in zip(
            similar_stack_ids, dissimilar_stack_ids
        ):
            test_data.append(
                get_data_row(coder, event, similar_stack_id, dissimilar_stack_id)
            )

    print(
        "Stack trace loaded for training and testing. Train:",
        len(train_data),
        "Test:",
        len(test_data),
    )
    return train_data, test_data


if generate_dataset:
    # Load from netbeans bucket
    all_train, all_test = generate_dataset_for_train_test(
        bucket_name,
        dataset_json,
        num_train_pairs,
        num_test_pairs,
        trim_length,
        test_only=test_only,
    )

    if not test_only:
        # all_train.extend(train_stacks)
        train_dataset = Dataset.from_list(all_train)
        train_dataset.save_to_disk(f"{datasets_dir}/{dataset_key}_train")

    # all_test.extend(test_stacks)
    test_dataset = Dataset.from_list(all_test)
    test_dataset.save_to_disk(f"{datasets_dir}/{dataset_key}_eval")

    # Delete lists to free up memory
    del all_train
    del all_test

# exit()
print("Load the preprocessed dataset")
test_dataset = Dataset.load_from_disk(f"{datasets_dir}/{dataset_key}_eval")

stack_formatter.format(test_dataset[0]["anchor"])

print("Before formatting", test_dataset[0])
test_dataset = test_dataset.map(format_data_row, num_proc=20)
print("After formatting", test_dataset[0])
eval_dataset = test_dataset.select(range(eval_size))
test_dataset = test_dataset.shuffle(seed=42)

# 4. Define a loss function
loss = MultipleNegativesRankingLoss(model)


# Populate the lists with the sentences and their labels from test_dataset
sentences_1 = []
sentences_2 = []
labels = []

for i in range(len(eval_dataset)):
    sentences_1.append(eval_dataset[i]["anchor"])
    sentences_2.append(eval_dataset[i]["positive"])
    labels.append(1)  # 1 means similar pair

    sentences_1.append(eval_dataset[i]["anchor"])
    sentences_2.append(eval_dataset[i]["negative"])
    labels.append(0)  # 0 means dissimilar pair


# 6. (Optional) Create an evaluator & evaluate the base model
dev_evaluator = TripletEvaluator(
    anchors=eval_dataset["anchor"],
    positives=eval_dataset["positive"],
    negatives=eval_dataset["negative"],
    name="all-nli-dev",
    show_progress_bar=True,
    batch_size=batch_size,
)

embedding_evaluator = EmbeddingSimilarityEvaluator(
    sentences1=sentences_1[:eval_size],
    sentences2=sentences_2[:eval_size],
    scores=labels[:eval_size],
    main_similarity="cosine",
    show_progress_bar=True,
    batch_size=batch_size,
)


# print("Evaluate the base model...")
res_embedding = embedding_evaluator(model)
print("Initial embedding evaluation result: ", res_embedding)

result = dev_evaluator(model)
print("Initial triplet evaluation result: ", result)

config_to_write["initial_embedding_eval_result"] = res_embedding
config_to_write["initial_triplet_eval_result"] = result

if test_only:
    exit()

train_dataset = Dataset.load_from_disk(f"{datasets_dir}/{dataset_key}_train")
train_dataset = train_dataset.shuffle(seed=42)
train_dataset = train_dataset.map(format_data_row, num_proc=20)
print("Dataset Sizes - Train:", len(train_dataset), "Test:", len(test_dataset))

# 5. (Optional) Specify training arguments
args = SentenceTransformerTrainingArguments(
    # Required parameter:
    output_dir=f"{model_dir}/{run_name}",
    # Optional training parameters:
    num_train_epochs=2,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    learning_rate=2e-5,
    warmup_ratio=0.1,
    fp16=True,  # Set to False if you get an error that your GPU can't run on FP16
    bf16=False,  # Set to True if you have a GPU that supports BF16
    batch_sampler=BatchSamplers.NO_DUPLICATES,  # MultipleNegativesRankingLoss benefits from no duplicate samples in a batch
    # Optional tracking/debugging parameters:
    eval_strategy="steps",
    eval_steps=500,
    save_strategy="steps",
    save_steps=500,
    save_total_limit=2,
    logging_steps=50,
    load_best_model_at_end=True,  # Enable saving the best model
    metric_for_best_model="eval_loss",  # Specify the metric to monitor
    greater_is_better=False,
    # run_name=run_name,  # Will be used in W&B if `wandb` is installed
)

# 7. Create a trainer & train
trainer = SentenceTransformerTrainer(
    model=model,
    args=args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    loss=loss,
    evaluator=[embedding_evaluator, dev_evaluator],
)
trainer.train()

# (Optional) Evaluate the trained model on the test set
test_evaluator = TripletEvaluator(
    anchors=test_dataset["anchor"],
    positives=test_dataset["positive"],
    negatives=test_dataset["negative"],
    name="all-nli-test",
    show_progress_bar=True,
)
test_embedding_evaluator = EmbeddingSimilarityEvaluator(
    sentences1=sentences_1,
    sentences2=sentences_2,
    scores=labels,
    main_similarity="cosine",
    show_progress_bar=True,
    batch_size=batch_size,
)

result = test_evaluator(model)
print("Final triplet test result: ", result)

res_embedding = test_embedding_evaluator(model)
print("Final embedding test result: ", res_embedding)

config_to_write["final_embedding_eval_result"] = res_embedding
config_to_write["final_triplet_eval_result"] = result

# 8. Save the trained model
model.save_pretrained(f"{config_to_write['model_save_path']}")

json.dump(
    config_to_write,
    open(
        f"{artifact_dir}/{run_name}_{time.time()}.json",
        "w",
    ),
    indent=2,
)
