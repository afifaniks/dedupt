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

model = SentenceTransformer("all-mpnet-base-v2")
print(f"Model sequence length: {model.max_seq_length}")
# model.max_seq_length = 1024

# Check trainable parameters
# for name, param in model.named_parameters():
#     if param.requires_grad:
#         print(name)
# print(model.encode(["Hello, World!"], convert_to_tensor=True).shape)

# Count number of trainable parameters
# num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
# print(f"Number of trainable parameters: {num_params}")
# exit()
# model.to(device)

# Find model size in CUDA
print(f"Model size: {torch.cuda.memory_allocated() / 1024 ** 3:.2f} GB")


dataset_key = "eclipse_pretrain"
run_name = "mpnet-base-eclipse"
generate_dataset = True
batch_size = 16
eval_size = 1000
num_frames = 8


def format_stack(stack):
    # Select last 10 frames
    stack = stack[-num_frames:]
    return "\n".join([f"{i+1}: {frame}" for i, frame in enumerate(stack)])


def get_data_row(coder, event, similar_stack_id, dissimilar_stack_id):
    return {
        "anchor": format_stack(coder(event.st_id, transformer=True)),
        "positive": format_stack(coder(similar_stack_id, transformer=True)),
        "negative": format_stack(coder(dissimilar_stack_id, transformer=True)),
    }


if generate_dataset:
    triplet_selector_train = RandomTripletSelector(4)
    triplet_selector_eval = RandomTripletSelector(1)
    print("Load bucket data...")
    bucket_data = OtherBucketData(
        "eclipse",
        "/home/mdafifal.mamun/research/S3M/dataset/EMSE_data/eclipse_2018/eclipse_stacktraces.json",
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

    stack2seq = Stack2Seq(cased=False, trim_len=0, sep=".")

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
                get_data_row(coder, event, similar_stack_id, dissimilar_stack_id)
            )

    for i, event in tqdm(enumerate(data_gen.test()), desc="Step"):
        similar_stack_ids, dissimilar_stack_ids = triplet_selector_eval(event)
        for similar_stack_id, dissimilar_stack_id in zip(
            similar_stack_ids, dissimilar_stack_ids
        ):
            test_data.append(
                get_data_row(coder, event, similar_stack_id, dissimilar_stack_id)
            )

    # Convert train_data list to a Dataset
    train_dataset = Dataset.from_list(train_data)
    test_dataset = Dataset.from_list(test_data)
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
    num_train_epochs=3,
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
    save_steps=100,
    save_total_limit=2,
    logging_steps=50,
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
