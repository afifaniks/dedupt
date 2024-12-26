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

# 1. Load a model to finetune with 2. (Optional) model card data
model = SentenceTransformer(
    "BAAI/bge-m3",  # Any model from https://www.sbert.net/docs/pretrained_models.html
    model_card_data=SentenceTransformerModelCardData(
        language="en",
        license="apache-2.0",
        model_name="MPNet base trained on AllNLI triplets",
    ),
    trust_remote_code=True,
)
dataset_key = "eclipse_pretrain"
run_name = "mpnet-base-eclipse"


# triplet_selector_train = RandomTripletSelector(4)
# triplet_selector_eval = RandomTripletSelector(3)
# bucket_data = OtherBucketData(
#     "eclipse",
#     "/home/mdafifal.mamun/research/S3M/dataset/EMSE_data/eclipse_2018/eclipse_stacktraces.json",
#     3850,
#     700,
#     350,
#     140,
# )
# bucket_data.load()
# stack_loader = bucket_data.stack_loader()
# data_gen = BucketDataset(bucket_data)
# unsup_stacks = data_gen.train_stacks()

# data_gen.reset()

# stack2seq = Stack2Seq(cased=False, trim_len=0, sep=".")

# coder = SeqCoder(stack_loader, stack2seq, SimpleTokenizer(), min_freq=0, max_len=None)

# coder.fit(unsup_stacks)

# train_data = []
# test_data = []

# for i, event in tqdm(enumerate(data_gen.train()), desc="Step"):
#     similar_stack_ids, dissimilar_stack_ids = triplet_selector_train(event)
#     for similar_stack_id, dissimilar_stack_id in zip(
#         similar_stack_ids, dissimilar_stack_ids
#     ):
#         train_data.append(
#             {
#                 "anchor": "\n".join(coder(event.st_id, transformer=True)),
#                 "positive": "\n".join(coder(similar_stack_id, transformer=True)),
#                 "negative": "\n".join(coder(dissimilar_stack_id, transformer=True)),
#             }
#         )

# for i, event in tqdm(enumerate(data_gen.test()), desc="Step"):
#     similar_stack_ids, dissimilar_stack_ids = triplet_selector_eval(event)
#     for similar_stack_id, dissimilar_stack_id in zip(
#         similar_stack_ids, dissimilar_stack_ids
#     ):
#         test_data.append(
#             {
#                 "anchor": "\n".join(coder(event.st_id, transformer=True)),
#                 "positive": "\n".join(coder(similar_stack_id, transformer=True)),
#                 "negative": "\n".join(coder(dissimilar_stack_id, transformer=True)),
#             }
#         )

# # Convert train_data list to a Dataset

# train_dataset = Dataset.from_list(train_data)
# test_dataset = Dataset.from_list(test_data)
# train_dataset.save_to_disk(f"datasets/{dataset_key}_train")
# test_dataset.save_to_disk(f"datasets/{dataset_key}_eval")


print("Load the preprocessed dataset")
train_dataset = Dataset.load_from_disk(f"datasets/{dataset_key}_train")
test_dataset = Dataset.load_from_disk(f"datasets/{dataset_key}_eval")


def plot_tsne(model, test_dataset, run_name, state):
    print("Plotting T-SNE...")
    anchors = model.encode(test_dataset["anchor"], convert_to_tensor=True)
    positives = model.encode(test_dataset["positive"], convert_to_tensor=True)
    negatives = model.encode(test_dataset["negative"], convert_to_tensor=True)

    # Combine all embeddings and reduce dimensions
    all_embeddings = torch.cat([anchors, positives, negatives])
    labels = (
        ["anchor"] * len(anchors)
        + ["positive"] * len(positives)
        + ["negative"] * len(negatives)
    )

    # Use T-SNE for dimensionality reduction
    tsne = TSNE(n_components=2, random_state=42)
    reduced_embeddings = tsne.fit_transform(all_embeddings.cpu().numpy())

    # Plot the reduced embeddings
    plt.figure(figsize=(10, 7))
    for label, color in zip(
        ["anchor", "positive", "negative"], ["blue", "green", "red"]
    ):
        indices = [i for i, lbl in enumerate(labels) if lbl == label]
        plt.scatter(
            reduced_embeddings[indices, 0],
            reduced_embeddings[indices, 1],
            label=label,
            alpha=0.7,
        )
    plt.title(f"T-SNE {run_name} - {state}")
    plt.legend()
    plt.savefig(f"tsne_{run_name} - {state}.png")
    plt.show()


# Optionally preprocess and shuffle the dataset
train_dataset = train_dataset.shuffle(seed=42)
test_dataset = test_dataset.shuffle(seed=42)

print("Dataset Sizes - Train:", len(train_dataset), "Test:", len(test_dataset))

# 4. Define a loss function
loss = MultipleNegativesRankingLoss(model)

# 5. (Optional) Specify training arguments
args = SentenceTransformerTrainingArguments(
    # Required parameter:
    output_dir=f"models/{run_name}",
    # Optional training parameters:
    num_train_epochs=2,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
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
    run_name=run_name,  # Will be used in W&B if `wandb` is installed
)

# 6. (Optional) Create an evaluator & evaluate the base model
dev_evaluator = TripletEvaluator(
    anchors=test_dataset["anchor"],
    positives=test_dataset["positive"],
    negatives=test_dataset["negative"],
    name="all-nli-dev",
)

sentences_1 = []
sentences_2 = []
labels = []

# Populate the lists with the sentences and their labels from test_dataset
for i in range(len(test_dataset)):
    sentences_1.append(test_dataset[i]["anchor"])
    sentences_2.append(test_dataset[i]["positive"])
    labels.append(1)  # 1 means similar pair

    sentences_1.append(test_dataset[i]["anchor"])
    sentences_2.append(test_dataset[i]["negative"])
    labels.append(0)  # 0 means dissimilar pair


embedding_evaluator = EmbeddingSimilarityEvaluator(
    sentences1=sentences_1,
    sentences2=sentences_2,
    scores=labels,
    main_similarity="cosine",
)

print("Evaluate the base model")
res_embedding = embedding_evaluator(model)
print("Initial embedding evaluation result: ", res_embedding)
exit()

result = dev_evaluator(model)
print("Initial evaluation result: ", result)

# Visualize the embeddings before training
plot_tsne(model, test_dataset, run_name, "after")

# 7. Create a trainer & train
trainer = SentenceTransformerTrainer(
    model=model,
    args=args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    loss=loss,
    evaluator=dev_evaluator,
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
test_evaluator(model)

# 8. Save the trained model
model.save_pretrained(f"models/{run_name}/final")

# 9. (Optional) Push it to the Hugging Face Hub
# model.push_to_hub("mpnet-base-all-nli-triplet")
