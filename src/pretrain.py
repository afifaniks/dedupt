import json

import matplotlib.pyplot as plt
import torch
from sentence_transformers import (
    SentenceTransformer,
    SentenceTransformerModelCardData,
    SentenceTransformerTrainer,
    SentenceTransformerTrainingArguments,
)
from sentence_transformers.evaluation import (
    EmbeddingSimilarityEvaluator,
    TripletEvaluator,
)
from sentence_transformers.losses import MultipleNegativesRankingLoss
from sentence_transformers.training_args import BatchSamplers
from sklearn.manifold import TSNE
from tqdm import tqdm

from data.buckets.bucket_data import BucketData, OtherBucketData
from data.buckets.issues_data import BucketDataset
from data.triplet_selector import RandomTripletSelector
from datasets import Dataset
from preprocess.entry_coders import Stack2Seq
from preprocess.seq_coder import SeqCoder
from preprocess.tokenizers import SimpleTokenizer

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

model = SentenceTransformer("BAAI/bge-base-en")
print(f"Model sequence length: {model.max_seq_length}")

# Find model size in CUDA
print(f"Model size: {torch.cuda.memory_allocated() / 1024 ** 3:.2f} GB")


dataset_key = "combined_pretrain_10"
run_name = "bge-base-combined_10"
generate_dataset = True
batch_size = 16
eval_size = 1000
num_frames = 10
num_train_pairs = 5
num_test_pairs = 1
# trim_length = 0
frame_freq = {}

# Print all training parameters
print("Train pair:", num_train_pairs, "Test pair:", num_test_pairs)


def format_stack(stack):
    # Remove duplicate frames
    stack = list(dict.fromkeys(stack))
    stack = stack[-num_frames:]
    return "\n".join([f"{i+1}: {frame}" for i, frame in enumerate(stack)])


def get_data_row(
    coder, event, similar_stack_id, dissimilar_stack_id, add_to_freq=False
):
    anchor_frames = coder(event.st_id, transformer=True)
    positive_frames = coder(similar_stack_id, transformer=True)
    negative_frames = coder(dissimilar_stack_id, transformer=True)

    if add_to_freq:
        for frames in [anchor_frames, positive_frames, negative_frames]:
            for frame in frames:
                frame_freq[frame] = frame_freq.get(frame, 0) + 1

    return {
        "anchor": format_stack(anchor_frames),
        "positive": format_stack(positive_frames),
        "negative": format_stack(negative_frames),
    }


def generate_dataset_for_train_test(
    bucket_name, dataset_path, num_train_pairs, num_test_pairs, trim_length
):
    triplet_selector_train = RandomTripletSelector(num_train_pairs)
    triplet_selector_eval = RandomTripletSelector(num_test_pairs)
    print("Load bucket data...")
    bucket_data = OtherBucketData(
        bucket_name,
        dataset_path,
        3850,
        700,
        350,
        140,
    )
    bucket_data.load()
    stack_loader = bucket_data.stack_loader()
    data_gen = BucketDataset(bucket_data)
    unsup_stacks = data_gen.train_stacks()

    data_gen.reset()

    stack2seq = Stack2Seq(cased=False, trim_len=trim_length, sep=".")

    coder = SeqCoder(
        stack_loader, stack2seq, SimpleTokenizer(), min_freq=0, max_len=None
    )

    coder.fit(unsup_stacks)

    train_data = []
    test_data = []

    print("Generate the dataset...")
    for i, event in tqdm(enumerate(data_gen.train()), desc="Step"):
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
                    add_to_freq=False,
                )
            )

    for i, event in tqdm(enumerate(data_gen.test()), desc="Step"):
        similar_stack_ids, dissimilar_stack_ids = triplet_selector_eval(event)
        for similar_stack_id, dissimilar_stack_id in zip(
            similar_stack_ids, dissimilar_stack_ids
        ):
            test_data.append(
                get_data_row(coder, event, similar_stack_id, dissimilar_stack_id)
            )

    return train_data, test_data


if generate_dataset:
    all_train = []
    all_test = []

    # Load from netbeans bucket
    nb_train, nb_test = generate_dataset_for_train_test(
        "netbeans",
        "/home/mdafifal.mamun/research/S3M/dataset/EMSE_data/netbeans_2016/netbeans_stacktraces.json",
        num_train_pairs,
        num_test_pairs,
        0,
    )
    ec_train, ec_test = generate_dataset_for_train_test(
        "eclipse",
        "/home/mdafifal.mamun/research/S3M/dataset/EMSE_data/eclipse_2018/eclipse_stacktraces.json",
        num_train_pairs,
        num_test_pairs,
        2,
    )

    all_train.extend(nb_train)
    all_train.extend(ec_train)
    all_test.extend(nb_test)
    all_test.extend(ec_test)

    # Convert train_data list to a Dataset
    train_dataset = Dataset.from_list(all_train)
    test_dataset = Dataset.from_list(all_test)
    train_dataset.save_to_disk(f"datasets/{dataset_key}_train")
    test_dataset.save_to_disk(f"datasets/{dataset_key}_eval")


print("Load the preprocessed dataset")
train_dataset = Dataset.load_from_disk(f"datasets/{dataset_key}_train")
test_dataset = Dataset.load_from_disk(f"datasets/{dataset_key}_eval")

# Optionally preprocess and shuffle the dataset
train_dataset = train_dataset.shuffle(seed=42)
test_dataset = test_dataset.shuffle(seed=42)
eval_dataset = test_dataset.select(range(eval_size))

print("Dataset Sizes - Train:", len(train_dataset), "Test:", len(test_dataset))

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


# 5. (Optional) Specify training arguments
args = SentenceTransformerTrainingArguments(
    # Required parameter:
    output_dir=f"models/{run_name}",
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
    eval_steps=200,
    save_strategy="steps",
    save_steps=200,
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


# 8. Save the trained model
model.save_pretrained(f"models/{run_name}/final")

# 9. (Optional) Push it to the Hugging Face Hub
# model.push_to_hub("mpnet-base-all-nli-triplet")
