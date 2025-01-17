import torch
import torch.nn.functional as F

torch.manual_seed(0)  # For reproducibility


def find_most_similar_pair(list1, list2):
    embeddings1 = torch.tensor(list1) if not isinstance(list1, torch.Tensor) else list1
    embeddings2 = torch.tensor(list2) if not isinstance(list2, torch.Tensor) else list2

    embeddings1 = F.normalize(embeddings1, p=2, dim=1)
    embeddings2 = F.normalize(embeddings2, p=2, dim=1)

    similarity_matrix = torch.mm(embeddings1, embeddings2.T)

    print(similarity_matrix)

    max_sim = similarity_matrix.max().item()
    max_idx = similarity_matrix.argmax().item()


    i, j = divmod(max_idx, similarity_matrix.shape[1])
    print(i, j)

    return max_sim, (i, j)


# Example embeddings (randomly initialized for illustration)
list1 = torch.rand(3, 768)  # 5 embeddings, each of size 768
list2 = torch.rand(2, 768)  # 6 embeddings, each of size 768

max_similarity, (index1, index2) = find_most_similar_pair(list1, list2)

print(
    f"Most similar pair: Embedding {index1} from List1 and Embedding {index2} from List2"
)
print(f"Maximum similarity: {max_similarity}")
