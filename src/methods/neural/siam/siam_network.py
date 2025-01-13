from typing import Iterable, List, Tuple

import torch
import torch.nn as nn

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

    def aggregate_embeddings(self, stack_id):
        embeddings = self.encoder.forward(stack_id)
        return torch.mean(embeddings, dim=0)

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
