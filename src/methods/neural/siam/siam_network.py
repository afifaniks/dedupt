from typing import Iterable, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from methods.neural.neural_base import NeuralModel
from methods.neural.siam.classifier import StackClassifier


class SiamMultiModalModel(NeuralModel):
    def __init__(self, encoders, agg, **kwargs):
        super(SiamMultiModalModel, self).__init__()
        self.agg = agg(encoders)
        self.classifier = StackClassifier(input_dim=self.agg.out_dim(), **kwargs)
        self.cache = {}

    def fit(
        self,
        sim_train_data: List[Tuple[int, int, int]] = None,
        unsup_data: Iterable[int] = None,
    ):
        pass

    def get_agg(self, stack_id):
        if self.training:
            self.cache = {}
            return self.agg(stack_id)
        else:
            if stack_id not in self.cache:
                self.cache[stack_id] = self.agg(stack_id)
            return self.cache[stack_id]

    def forward(self, stack_id1, stack_id2):
        return self.classifier(self.get_agg(stack_id1), self.get_agg(stack_id2))

    def predict(self, anchor_id, stack_ids):
        with torch.no_grad():
            y_pr = []
            for stack_id in stack_ids:
                y_pr.append(self.forward(anchor_id, stack_id).cpu().numpy()[1])
            return y_pr

    def name(self):
        return self.agg.name() + "_siam_" + self.classifier.name()

    def train(self, mode=True):
        super().train(mode)

    def opt_params(self):
        return self.agg.opt_params() + self.classifier.opt_params()


class SiamSentTransformerModel(NeuralModel):
    def __init__(self, encoder, **kwargs):
        super(SiamSentTransformerModel, self).__init__()
        self.encoder = encoder
        self.classifier = StackClassifier(input_dim=self.encoder.out_dim(), **kwargs)
        self.cache = {}

    def fit(
        self,
        sim_train_data: List[Tuple[int, int, int]] = None,
        unsup_data: Iterable[int] = None,
    ):
        pass

    def get_agg(self, stack_id):
        if self.training:
            self.cache = {}
            return self.encoder(stack_id)
        else:
            if stack_id not in self.cache:
                self.cache[stack_id] = self.encoder(stack_id)
            return self.cache[stack_id]

    def forward(self, stack_id1, stack_id2):
        return self.classifier(
            self.encoder.forward(stack_id1), self.encoder.forward(stack_id2)
        )

    def predict(self, anchor_id, stack_ids):
        with torch.no_grad():
            y_pr = []
            for stack_id in stack_ids:
                y_pr.append(self.forward(anchor_id, stack_id).cpu().numpy()[1])
            return y_pr

    def name(self):
        return self.encoder.name() + "_senttrans_" + self.classifier.name()

    def train(self, mode=True):
        super().train(mode)

    def opt_params(self):
        return self.encoder.opt_params() + self.classifier.opt_params()


class SiamSentTransformerModelMultiStack(NeuralModel):
    def __init__(self, encoder, hidden_dim=256, **kwargs):
        super(SiamSentTransformerModelMultiStack, self).__init__()
        self.encoder = encoder
        self.hidden_dim = hidden_dim
        # self.fc = nn.Linear(self.encoder.out_dim(), self.hidden_dim)
        self.classifier = StackClassifier(
            input_dim=self.encoder.out_dim() * 2, **kwargs
        )
        self.alpha = nn.Parameter(torch.tensor(0.5))
        self.cache = {}
        print("Current model name:", self.name())

    def fit(
        self,
        sim_train_data: List[Tuple[int, int, int]] = None,
        unsup_data: Iterable[int] = None,
    ):
        pass

    def get_agg(self, stack_id):
        if self.training:
            self.cache = {}
            return self.encoder(stack_id)
        else:
            if stack_id not in self.cache:
                self.cache[stack_id] = self.encoder(stack_id)
            return self.cache[stack_id]

    def aggregate_embeddings(self, stack_id):
        embeddings = self.encoder.forward(stack_id)
        return embeddings  # torch.mean(embeddings, dim=0)

    def find_most_similar_pair(self, embeddings1, embeddings2):
        embeddings1 = F.normalize(embeddings1, p=2, dim=1)
        embeddings2 = F.normalize(embeddings2, p=2, dim=1)

        similarity_matrix = torch.mm(embeddings1, embeddings2.T)

        max_sim = similarity_matrix.max().item()
        max_idx = similarity_matrix.argmax().item()

        i, j = divmod(max_idx, similarity_matrix.shape[1])

        return max_sim, (i, j)

    def forward(self, stack_ids1, stack_ids2):
        agg_embedding1 = self.aggregate_embeddings(stack_ids1)
        agg_embedding2 = self.aggregate_embeddings(stack_ids2)
        max_sim, (i, j) = self.find_most_similar_pair(agg_embedding1, agg_embedding2)

        mean_agg_embedding1 = torch.mean(agg_embedding1, dim=0)
        mean_agg_embedding2 = torch.mean(agg_embedding2, dim=0)

        max_agg_embedding1 = agg_embedding1[i]
        max_agg_embedding2 = agg_embedding2[j]

        concat_agg_embedding1 = torch.concat(
            (self.alpha * max_agg_embedding1, (1 - self.alpha) * mean_agg_embedding1),
            dim=0,
        )
        concat_agg_embedding2 = torch.concat(
            (self.alpha * max_agg_embedding2, (1 - self.alpha) * mean_agg_embedding2),
            dim=0,
        )

        return self.classifier(concat_agg_embedding1, concat_agg_embedding2)

    def predict(self, anchor_id, stack_ids):
        with torch.no_grad():
            y_pr = []
            # anchor_embedding = self.aggregate_embeddings(anchor_id)
            for stack_id in stack_ids:
                y_pr.append(self.forward(anchor_id, stack_id).cpu().numpy()[1])
            return y_pr

    def name(self):
        return self.encoder.name() + "_senttrans"

    def train(self, mode=True):
        super().train(mode)

    def opt_params(self):
        return (
            self.encoder.opt_params()
            + list(self.fc.parameters())
            + self.classifier.opt_params()
        )


class SiamSentTransformerModelMultiStackAttention(NeuralModel):
    def __init__(self, encoder, hidden_dim=256, **kwargs):
        super(SiamSentTransformerModelMultiStackAttention, self).__init__()
        self.encoder = encoder
        self.hidden_dim = hidden_dim
        self.key_dim = hidden_dim // 2
        # self.fc = nn.Linear(self.encoder.out_dim(), self.hidden_dim)
        self.query_layer = nn.Linear(self.encoder.out_dim(), self.key_dim)
        self.key_layer = nn.Linear(self.encoder.out_dim(), self.key_dim)
        self.value_layer = nn.Linear(self.encoder.out_dim(), self.hidden_dim)
        self.classifier = StackClassifier(input_dim=self.hidden_dim, **kwargs)
        self.cache = {}

    def fit(
        self,
        sim_train_data: List[Tuple[int, int, int]] = None,
        unsup_data: Iterable[int] = None,
    ):
        pass

    def get_agg(self, stack_id):
        if self.training:
            self.cache = {}
            return self.encoder(stack_id)
        else:
            if stack_id not in self.cache:
                self.cache[stack_id] = self.encoder(stack_id)
            return self.cache[stack_id]

    def attention_aggregation(self, embeddings):
        """
        Use scaled dot-product attention to combine embeddings dynamically.
        :param embeddings: Tensor of shape (num_stack_traces, embedding_dim)
        :return: Aggregated embedding of shape (embedding_dim,)
        """
        # Query, Key, Value transformations
        queries = self.query_layer(embeddings)  # (num_stack_traces, key_dim)
        keys = self.key_layer(embeddings)  # (num_stack_traces, key_dim)
        values = self.value_layer(embeddings)  # (num_stack_traces, hidden_dim)

        # Compute attention scores: (num_stack_traces, num_stack_traces)
        scores = torch.matmul(queries, keys.transpose(-2, -1)) / (self.key_dim**0.5)
        attention_weights = F.softmax(scores, dim=-1)  # Normalize scores

        # Aggregate embeddings: (num_stack_traces, hidden_dim) -> (hidden_dim,)
        aggregated_embedding = torch.matmul(attention_weights, values).sum(dim=0)

        return aggregated_embedding

    def aggregate_embeddings(self, stack_id):
        embeddings = self.encoder.forward(stack_id)
        return self.attention_aggregation(embeddings)

    def forward(self, stack_ids1, stack_ids2):
        agg_embedding1 = self.aggregate_embeddings(stack_ids1)
        agg_embedding2 = self.aggregate_embeddings(stack_ids2)

        # Pass through the fully connected layer
        # agg_embedding1 = self.fc(anchor_embedding)
        # agg_embedding2 = self.fc(agg_embedding2)
        return self.classifier(agg_embedding1, agg_embedding2)

    def predict(self, anchor_id, stack_ids):
        with torch.no_grad():
            y_pr = []
            # anchor_embedding = self.aggregate_embeddings(anchor_id)
            for stack_id in stack_ids:
                y_pr.append(self.forward(anchor_id, stack_id).cpu().numpy()[1])
            return y_pr

    def name(self):
        return self.encoder.name() + "_senttrans_" + self.classifier.name()

    def train(self, mode=True):
        super().train(mode)

    def opt_params(self):
        return (
            self.encoder.opt_params()
            + list(self.fc.parameters())
            + self.classifier.opt_params()
        )


class SiamSentTransformerModelMultiStackMultiHead(NeuralModel):
    def __init__(self, encoder, hidden_dim=512, num_heads=12, **kwargs):
        super(SiamSentTransformerModelMultiStackMultiHead, self).__init__()
        self.encoder = encoder
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads

        # Fully connected layer for optional transformation after aggregation
        # self.fc = nn.Linear(self.encoder.out_dim(), self.hidden_dim)

        # Multi-head attention to aggregate stack trace embeddings
        self.attention = nn.MultiheadAttention(
            embed_dim=self.encoder.out_dim(), num_heads=num_heads
        )

        # Classifier for duplicate prediction
        self.classifier = StackClassifier(input_dim=self.encoder.out_dim(), **kwargs)
        self.cache = {}

    def fit(
        self,
        sim_train_data: List[Tuple[int, int, int]] = None,
        unsup_data: Iterable[int] = None,
    ):
        pass

    def get_agg(self, stack_id):
        if self.training:
            self.cache = {}
            return self.encoder(stack_id)
        else:
            if stack_id not in self.cache:
                self.cache[stack_id] = self.encoder(stack_id)
            return self.cache[stack_id]

    def aggregate_embeddings(self, stack_ids1, stack_ids2):
        # # Encode stack traces
        # embeddings = self.encoder.forward(
        #     stack_ids
        # )  # Shape: (num_stacks, embedding_dim)

        # # Add batch dimension for attention compatibility
        # embeddings = embeddings.unsqueeze(1)  # Shape: (num_stacks, 1, embedding_dim)

        # # Apply multi-head attention (query, key, value all set to embeddings for self-attention)
        # attn_output, _ = self.attention(embeddings, embeddings, embeddings)
        # attn_output = attn_output.squeeze(1)  # Shape: (num_stacks, embedding_dim)

        # # Aggregate the attention-weighted embeddings
        # aggregated_embedding = attn_output.mean(dim=0)  # Shape: (embedding_dim,)

        # # Optional: Pass through a fully connected layer
        # return aggregated_embedding

        # Encode stack traces for both bug reports
        embeddings1 = self.encoder.forward(
            stack_ids1
        )  # Shape: (num_stacks1, embedding_dim)
        embeddings2 = self.encoder.forward(
            stack_ids2
        )  # Shape: (num_stacks2, embedding_dim)

        # if len(embeddings1) > 1 or len(embeddings2) > 1:
        #     print("break")

        # Add batch dimension for attention compatibility
        # embeddings1 = embeddings1.permute(
        #     1, 0, 2
        # )  # Shape: (num_stacks1, batch_size=1, embedding_dim)
        # embeddings2 = embeddings2.permute(
        #     1, 0, 2
        # )  # Shape: (num_stacks2, batch_size=1, embedding_dim)

        # Apply cross-attention: embeddings1 attends to embeddings2, and vice versa
        attn_output1, attn_weights1 = self.attention(
            embeddings1, embeddings2, embeddings2
        )
        attn_output2, attn_weights2 = self.attention(
            embeddings2, embeddings1, embeddings1
        )

        # Use attention weights to weight the attention output (weighted aggregation)
        agg_embedding1 = attn_output1.max(
            dim=0
        ).values  # Aggregate across stack traces (Shape: (batch_size, embedding_dim))
        agg_embedding2 = attn_output2.max(dim=0).values

        return agg_embedding1, agg_embedding2

    def forward(self, stack_ids1, stack_ids2):
        # Aggregate embeddings using MHA
        # agg_embedding1 = self.aggregate_embeddings(stack_ids1)
        # agg_embedding2 = self.aggregate_embeddings(stack_ids2)
        agg_embedding1, agg_embedding2 = self.aggregate_embeddings(
            stack_ids1, stack_ids2
        )

        # Classification step
        return self.classifier(agg_embedding1, agg_embedding2)

    def predict(self, anchor_id, stack_ids):
        with torch.no_grad():
            y_pr = []
            for stack_id in stack_ids:
                y_pr.append(self.forward(anchor_id, stack_id).cpu().numpy()[1])
            return y_pr

    def name(self):
        return self.encoder.name() + "_senttrans_" + self.classifier.name()

    def train(self, mode=True):
        super().train(mode)

    def opt_params(self):
        return (
            self.encoder.opt_params()
            + list(self.fc.parameters())
            + list(self.attention.parameters())
            + self.classifier.opt_params()
        )
