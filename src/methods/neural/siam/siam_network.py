from typing import Iterable, List, Tuple

import torch

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

    def predictx(self, anchor_id, stack_ids):  # 2D List of stack IDs
        with torch.no_grad():
            # Ensure anchor_id is processed only once
            anchor_embedding = self.encoder.forward(anchor_id).unsqueeze(
                0
            )  # Add batch dimension

            # Flatten and keep track of mapping to restore structure
            flat_list = []
            id_map = []
            for key, sublist in stack_ids.items():
                flat_list.extend(sublist)
                id_map.extend([key] * len(sublist))

            # Forward pass through encoder
            stack_embeddings = self.encoder.forward_all(flat_list)

            # Generate predictions
            predictions = (
                self.classifier(anchor_embedding, stack_embeddings).cpu().numpy()
            )

            # Group predictions by original keys
            preds = {}
            for key, pred in zip(id_map, predictions[:, 1]):
                if key not in preds:
                    preds[key] = []
                preds[key].append(pred)

            return preds

    def name(self):
        return self.encoder.name() + "_senttrans_" + self.classifier.name()

    def train(self, mode=True):
        super().train(mode)

    def opt_params(self):
        return self.encoder.opt_params() + self.classifier.opt_params()
