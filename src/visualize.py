import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from sentence_transformers import SentenceTransformer
from sklearn.manifold import TSNE
from sklearn.metrics.pairwise import cosine_distances

from datasets import Dataset

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

dataset_key = "netbeans_pretrain_10"
run_name = "bge-base-netbeans_10"
generate_dataset = True
batch_size = 16
eval_size = 1000
num_frames = 10
num_train_pairs = 5
num_test_pairs = 1
# trim_length = 0
frame_freq = {}

test_dataset = Dataset.load_from_disk(f"datasets/{dataset_key}_eval")
model = SentenceTransformer("BAAI/bge-base-en")
model_after = SentenceTransformer(f"models/{run_name}/final")

print(f"Model sequence length: {model.max_seq_length}")

# Find model size in CUDA
print(f"Model size: {torch.cuda.memory_allocated() / 1024 ** 3:.2f} GB")


def visualize_distances_before_after(
    test_dataset, model_before, model_after, num_samples=10, metric="cosine"
):
    samples = test_dataset.select(range(num_samples))

    anchor_positive_distances_before = []
    anchor_negative_distances_before = []
    anchor_positive_distances_after = []
    anchor_negative_distances_after = []

    # Compute distances for each triplet
    for sample in samples:
        anchor, positive, negative = (
            sample["anchor"],
            sample["positive"],
            sample["negative"],
        )

        # Embeddings before training
        emb_anchor_before = model_before.encode(anchor)
        emb_positive_before = model_before.encode(positive)
        emb_negative_before = model_before.encode(negative)

        # Embeddings after training
        emb_anchor_after = model_after.encode(anchor)
        emb_positive_after = model_after.encode(positive)
        emb_negative_after = model_after.encode(negative)

        # Compute distances
        if metric == "cosine":
            anchor_positive_distances_before.append(
                cosine_distances([emb_anchor_before], [emb_positive_before])[0][0]
            )
            anchor_negative_distances_before.append(
                cosine_distances([emb_anchor_before], [emb_negative_before])[0][0]
            )
            anchor_positive_distances_after.append(
                cosine_distances([emb_anchor_after], [emb_positive_after])[0][0]
            )
            anchor_negative_distances_after.append(
                cosine_distances([emb_anchor_after], [emb_negative_after])[0][0]
            )
        elif metric == "euclidean":
            anchor_positive_distances_before.append(
                np.linalg.norm(emb_anchor_before - emb_positive_before)
            )
            anchor_negative_distances_before.append(
                np.linalg.norm(emb_anchor_before - emb_negative_before)
            )
            anchor_positive_distances_after.append(
                np.linalg.norm(emb_anchor_after - emb_positive_after)
            )
            anchor_negative_distances_after.append(
                np.linalg.norm(emb_anchor_after - emb_negative_after)
            )

    # Create plots
    fig, axes = plt.subplots(1, 2, figsize=(16, 6), sharey=True)

    # Plot distances before training
    axes[0].scatter(
        range(num_samples),
        anchor_positive_distances_before,
        color="green",
        label="Anchor-Positive",
        alpha=0.7,
    )
    axes[0].scatter(
        range(num_samples),
        anchor_negative_distances_before,
        color="red",
        label="Anchor-Negative",
        alpha=0.7,
    )
    axes[0].set_title("Distances Before Training")
    axes[0].set_xlabel("Sample Index")
    axes[0].set_ylabel(f"{metric.capitalize()} Distance")
    axes[0].legend()
    axes[0].grid()

    # Plot distances after training
    axes[1].scatter(
        range(num_samples),
        anchor_positive_distances_after,
        color="green",
        label="Anchor-Positive",
        alpha=0.7,
    )
    axes[1].scatter(
        range(num_samples),
        anchor_negative_distances_after,
        color="red",
        label="Anchor-Negative",
        alpha=0.7,
    )
    axes[1].set_title("Distances After Training")
    axes[1].set_xlabel("Sample Index")
    axes[1].legend()
    axes[1].grid()

    plt.tight_layout()
    plt.savefig("distance_plot_base.png")
    plt.show()


import matplotlib.pyplot as plt
import numpy as np
from sklearn.manifold import TSNE


def visualize_distances_before_after_2(
    test_dataset, model_before, model_after, num_samples=10, metric="cosine"
):
    # Select samples from the dataset
    samples = test_dataset.select(range(num_samples))

    # Initialize lists to store embeddings before and after training
    embeddings_before = []
    embeddings_after = []
    labels = []

    # Generate embeddings before and after training for anchor, positive, and negative samples
    for i, sample in enumerate(samples):
        anchor, positive, negative = (
            sample["anchor"],
            sample["positive"],
            sample["negative"],
        )

        # Generate a unique label for each triplet
        triplet_label = f"Triplet {i+1}"

        # Encode using both models
        emb_anchor_before = model_before.encode(anchor)
        emb_positive_before = model_before.encode(positive)
        emb_negative_before = model_before.encode(negative)

        emb_anchor_after = model_after.encode(anchor)
        emb_positive_after = model_after.encode(positive)
        emb_negative_after = model_after.encode(negative)

        # Append embeddings and labels to lists
        embeddings_before.append(
            [emb_anchor_before, emb_positive_before, emb_negative_before]
        )
        embeddings_after.append(
            [emb_anchor_after, emb_positive_after, emb_negative_after]
        )
        labels.append(triplet_label)

    # Flatten the embeddings list for TSNE
    embeddings_before = np.array(embeddings_before).reshape(
        -1, embeddings_before[0][0].shape[0]
    )
    embeddings_after = np.array(embeddings_after).reshape(
        -1, embeddings_after[0][0].shape[0]
    )

    # Apply t-SNE to reduce dimensionality to 2D
    tsne = TSNE(n_components=2, random_state=42)
    embeddings_before_2d = tsne.fit_transform(embeddings_before)
    embeddings_after_2d = tsne.fit_transform(embeddings_after)

    # Create the plot
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Plot Before Training
    for i, label in enumerate(labels):
        # Extract coordinates for anchor, positive, and negative points
        emb_anchor_before_2d = embeddings_before_2d[i]
        emb_positive_before_2d = embeddings_before_2d[num_samples + i]
        emb_negative_before_2d = embeddings_before_2d[2 * num_samples + i]

        # Plot anchor, positive, and negative points
        axes[0].scatter(
            emb_anchor_before_2d[0],
            emb_anchor_before_2d[1],
            color="blue",
            label="Anchor" if i == 0 else "",
            alpha=0.7,
            marker="o",
        )
        axes[0].scatter(
            emb_positive_before_2d[0],
            emb_positive_before_2d[1],
            color="green",
            label="Positive" if i == 0 else "",
            alpha=0.7,
            marker="^",
        )
        axes[0].scatter(
            emb_negative_before_2d[0],
            emb_negative_before_2d[1],
            color="red",
            label="Negative" if i == 0 else "",
            alpha=0.7,
            marker="s",
        )

        # Connect anchor to positive and negative
        axes[0].plot(
            [emb_anchor_before_2d[0], emb_positive_before_2d[0]],
            [emb_anchor_before_2d[1], emb_positive_before_2d[1]],
            color="green",
            alpha=0.7,
            linewidth=0.7,
        )
        axes[0].plot(
            [emb_anchor_before_2d[0], emb_negative_before_2d[0]],
            [emb_anchor_before_2d[1], emb_negative_before_2d[1]],
            color="red",
            alpha=0.7,
            linewidth=0.7,
        )

        # Annotate each triplet with the label
        axes[0].text(
            emb_anchor_before_2d[0],
            emb_anchor_before_2d[1],
            label,
            fontsize=9,
            alpha=0.7,
        )

    axes[0].set_title("Before Training")
    axes[0].set_xlabel("TSNE Component 1")
    axes[0].set_ylabel("TSNE Component 2")
    axes[0].legend()
    axes[0].grid()

    # Plot After Training
    for i, label in enumerate(labels):
        # Extract coordinates for anchor, positive, and negative points
        emb_anchor_after_2d = embeddings_after_2d[i]
        emb_positive_after_2d = embeddings_after_2d[num_samples + i]
        emb_negative_after_2d = embeddings_after_2d[2 * num_samples + i]

        # Plot anchor, positive, and negative points
        axes[1].scatter(
            emb_anchor_after_2d[0],
            emb_anchor_after_2d[1],
            color="blue",
            label="Anchor" if i == 0 else "",
            alpha=0.7,
            marker="o",
        )
        axes[1].scatter(
            emb_positive_after_2d[0],
            emb_positive_after_2d[1],
            color="green",
            label="Positive" if i == 0 else "",
            alpha=0.7,
            marker="^",
        )
        axes[1].scatter(
            emb_negative_after_2d[0],
            emb_negative_after_2d[1],
            color="red",
            label="Negative" if i == 0 else "",
            alpha=0.7,
            marker="s",
        )

        # Connect anchor to positive and negative
        axes[1].plot(
            [emb_anchor_after_2d[0], emb_positive_after_2d[0]],
            [emb_anchor_after_2d[1], emb_positive_after_2d[1]],
            color="green",
            alpha=0.7,
            linewidth=0.7,
        )
        axes[1].plot(
            [emb_anchor_after_2d[0], emb_negative_after_2d[0]],
            [emb_anchor_after_2d[1], emb_negative_after_2d[1]],
            color="red",
            alpha=0.7,
            linewidth=0.7,
        )

        # Annotate each triplet with the label
        axes[1].text(
            emb_anchor_after_2d[0], emb_anchor_after_2d[1], label, fontsize=9, alpha=0.7
        )

    axes[1].set_title("After Training")
    axes[1].set_xlabel("TSNE Component 1")
    axes[1].set_ylabel("TSNE Component 2")
    axes[1].legend()
    axes[1].grid()

    plt.tight_layout()
    plt.savefig("embedding_movement_with_connections.png")
    plt.show()


def generate_tsne_plot(test_dataset, model, num_samples=10):
    # Extract the first 'num_samples' test samples
    samples = test_dataset.select(range(num_samples))

    # Prepare lists to hold embeddings and labels for anchor, positive, and negative samples
    embeddings = []
    labels = []  # 0 for anchor, 1 for positive, 2 for negative
    colors = ["blue", "green", "red"]  # colors for anchor, positive, negative

    for i in range(len(samples)):
        # Get anchor, positive, and negative sentences
        anchor_sentence = samples[i]["anchor"]
        positive_sentence = samples[i]["positive"]
        negative_sentence = samples[i]["negative"]

        # Get embeddings for anchor, positive, and negative
        anchor_embedding = model.encode([anchor_sentence])[0]
        positive_embedding = model.encode([positive_sentence])[0]
        negative_embedding = model.encode([negative_sentence])[0]

        # Append embeddings and labels
        embeddings.append(anchor_embedding)
        labels.append(0)  # Anchor

        embeddings.append(positive_embedding)
        labels.append(1)  # Positive

        embeddings.append(negative_embedding)
        labels.append(2)  # Negative

    embeddings = np.array(embeddings)
    labels = np.array(labels)

    # Apply t-SNE to reduce dimensionality to 2D for visualization
    tsne = TSNE(n_components=2, random_state=42)
    tsne_results = tsne.fit_transform(embeddings)

    # Plot the t-SNE results
    plt.figure(figsize=(8, 6))

    # Plot each type of sample with different colors
    for i, color in enumerate(colors):
        plt.scatter(
            tsne_results[labels == i, 0],
            tsne_results[labels == i, 1],
            c=color,
            label=["Anchor", "Positive", "Negative"][i],
            alpha=0.7,
        )

    plt.title("t-SNE Visualization of Anchor, Positive, and Negative Samples")
    plt.xlabel("t-SNE Component 1")
    plt.ylabel("t-SNE Component 2")
    plt.legend()
    plt.savefig("tsne_plot_base.png")
    plt.show()


# Example usage:
visualize_distances_before_after(test_dataset, model, model_after, num_samples=1500)


# Example usage
# generate_tsne_plot(test_dataset, model, num_samples=50)
