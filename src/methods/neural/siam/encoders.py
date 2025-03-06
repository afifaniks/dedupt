import time
import warnings
from abc import ABC, abstractmethod
from functools import lru_cache
from pprint import pprint
from timeit import timeit
from typing import List

import torch
from sentence_transformers import SentenceTransformer
from torch import nn

from data.formatters import StackFormatter
from methods.neural import device
from preprocess.seq_coder import SeqCoder


class Encoder(ABC, nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()

    @abstractmethod
    def out_dim(self) -> int:
        raise NotImplementedError

    @abstractmethod
    def opt_params(self) -> List[torch.tensor]:
        return []

    @abstractmethod
    def name(self) -> str:
        raise NotImplementedError


class EncoderModel(Encoder):
    def __init__(
        self, name: str, coder: SeqCoder, dim: int, out_dim: int = None, **kvargs
    ):
        super(EncoderModel, self).__init__()
        self.coder = coder
        self.word_embeddings = nn.Embedding(len(coder), dim)
        self.dim = dim
        self._out_dim = out_dim or dim
        self._name = coder.name() + "_" + name + f"_rand_dim={dim}"

    def to_inds(self, stack_id: int, reverse: bool = False) -> torch.tensor:
        if reverse:
            return torch.tensor(self.coder(stack_id)[::-1]).to(device)
        return torch.tensor(self.coder(stack_id)).to(device)

    def out_dim(self) -> int:
        return self._out_dim

    def opt_params(self) -> List[torch.tensor]:
        return []

    def name(self) -> str:
        return self._name

    def train(self, mode: bool = True):
        super().train(mode)
        self.coder.train(mode)


class LSTMEncoder(EncoderModel):
    def __init__(
        self,
        coder: SeqCoder,
        dim: int = 50,
        hid_dim: int = 200,
        bidir: bool = True,
        **kvargs,
    ):
        super(LSTMEncoder, self).__init__(
            f"lstm_frames.bidir={bidir},hdim={hid_dim}",
            coder,
            dim,
            out_dim=hid_dim,
            **kvargs,
        )
        self.hidden_dim = hid_dim // 2 if bidir else hid_dim
        self.bidir = bidir
        self.lstm_forward = nn.LSTM(dim, self.hidden_dim)
        self.lstm_backward = nn.LSTM(dim, self.hidden_dim)

    def forward(self, stack_id: int) -> torch.tensor:
        emb_f = self.word_embeddings(self.to_inds(stack_id))
        lstm_f_out, hidden_f = self.lstm_forward(emb_f.view(emb_f.shape[0], 1, -1))
        out = lstm_f_out[-1][0]
        if self.bidir:
            emb_b = self.word_embeddings(self.to_inds(stack_id, reverse=True))
            lstm_b_out, hidden_b = self.lstm_backward(emb_b.view(emb_b.shape[0], 1, -1))
            out = torch.cat((out, lstm_b_out[-1][0]))
        return out

    def opt_params(self) -> List[torch.tensor]:
        return list(self.lstm_forward.parameters()) + (
            list(self.lstm_backward.parameters()) if self.bidir else []
        )


class TransformerEncoder:
    def __init__(
        self,
        coder: SeqCoder,
        stack_formatter: StackFormatter,
        model_name: str,
        out_dim: int = 768,
        multi_stack: bool = False,
    ):
        # super(TransformerEncoder, self).__init__(
        #     f"transformer_{model_name}", None, dim=384, out_dim=out_dim, **kvargs
        # )
        super(TransformerEncoder, self).__init__()
        print(f"Loading transformer model: {model_name}")
        self.transformer = SentenceTransformer(model_name)
        print(f"Selected formatter: {stack_formatter.name()}")
        self.stack_formatter = stack_formatter
        self.output_dim = out_dim
        self.coder = coder
        self.multi_stack = multi_stack
        self._name = model_name.split("/")[-2].strip()

    @lru_cache(maxsize=200_000)
    def forward(self, stack_id: int) -> torch.Tensor:
        frames = self.coder(stack_id, transformer=True)
        if self.multi_stack:
            frames = [self.stack_formatter.format(frame) for frame in frames]
        else:
            frames = self.stack_formatter.format(frames)
        emb = self.transformer.encode(frames, convert_to_tensor=True)

        return emb

    def opt_params(self) -> list:
        return list(self.transformer.parameters())

    def out_dim(self) -> int:
        return self.output_dim

    def name(self) -> str:
        return self._name

    # Make the method deprecated
    def format_stack(self, stack):
        warnings.warn(
            "The 'format_stack' method is deprecated and will be removed",
            DeprecationWarning,
            stacklevel=2,
        )
        # Select last 10 frames
        stack = list(dict.fromkeys(stack))
        stack = [frame for frame in stack if frame.lower() != "none"]
        return "\n".join([f"{i+1}: {frame}" for i, frame in enumerate(stack)])

    # @lru_cache(maxsize=200_000)
    def forward_all(self, stack_ids: list) -> torch.Tensor:
        # Process all stack_ids in batch
        frames_batch = [
            self.coder(stack_id, transformer=True) for stack_id in stack_ids
        ]

        # Join frames for each stack_id into a single string
        sentences = ["\n".join(frames) for frames in frames_batch]

        # Encode the entire batch of sentences at once
        embeddings = self.transformer.encode(sentences, convert_to_tensor=True)

        return embeddings


class TrainableTransformerEncoder(nn.Module):
    def __init__(
        self,
        coder: SeqCoder,
        stack_formatter: StackFormatter,
        model_name: str,
        out_dim: int = 768,
        multi_stack: bool = False,
        enable_caching: bool = False,
    ):
        # super(TransformerEncoder, self).__init__(
        #     f"transformer_{model_name}", None, dim=384, out_dim=out_dim, **kvargs
        # )
        super(TrainableTransformerEncoder, self).__init__()
        print(f"Loading transformer model: {model_name}")
        self.transformer = SentenceTransformer(model_name)
        print(f"Selected formatter: {stack_formatter.name()}")
        self.stack_formatter = stack_formatter
        self.output_dim = out_dim
        self.coder = coder
        self.multi_stack = multi_stack
        self.cache_enabled = enable_caching
        self.cache = {}
        self._name = (
            model_name.split("/")[-2].strip() if "/" in model_name else model_name
        )
        self._name = f"{self._name}_trainable_encoder"

    def forward(self, stack_id: int) -> torch.Tensor:
        if self.cache_enabled and self.cache.get(stack_id) is not None:
            return self.cache[stack_id]

        frames = self.coder(stack_id, transformer=True)
        if self.multi_stack:
            frames = [self.stack_formatter.format(frame) for frame in frames]
        else:
            frames = self.stack_formatter.format(frames)
        emb = self.transformer.encode(frames, convert_to_tensor=True)

        if self.cache_enabled:
            self.cache[stack_id] = emb

        return emb

    def opt_params(self) -> list:
        return list(self.transformer.parameters())

    def out_dim(self) -> int:
        return self.output_dim

    def name(self) -> str:
        return self._name

    def clear_cache(self):
        self.cache = {}

    # Make the method deprecated
    def format_stack(self, stack):
        warnings.warn(
            "The 'format_stack' method is deprecated and will be removed",
            DeprecationWarning,
            stacklevel=2,
        )
        # Select last 10 frames
        stack = list(dict.fromkeys(stack))
        stack = [frame for frame in stack if frame.lower() != "none"]
        return "\n".join([f"{i+1}: {frame}" for i, frame in enumerate(stack)])

    # @lru_cache(maxsize=200_000)
    def forward_all(self, stack_ids: list) -> torch.Tensor:
        # Process all stack_ids in batch
        frames_batch = [
            self.coder(stack_id, transformer=True) for stack_id in stack_ids
        ]

        # Join frames for each stack_id into a single string
        sentences = ["\n".join(frames) for frames in frames_batch]

        # Encode the entire batch of sentences at once
        embeddings = self.transformer.encode(sentences, convert_to_tensor=True)

        return embeddings
    
    def enable_cache(self):
        print("Enabling cache...")
        self.cache_enabled = True
    
    def disable_cache(self):
        print("Disabling cache...")
        self.cache_enabled = False
        print("Clearing cache...")
        self.clear_cache()


class TransformerFrameEncoder:
    def __init__(
        self,
        coder: SeqCoder,
        stack_formatter: StackFormatter,
        model_name: str,
        out_dim: int = 768,
        multi_stack: bool = False,
    ):
        # super(TransformerEncoder, self).__init__(
        #     f"transformer_{model_name}", None, dim=384, out_dim=out_dim, **kvargs
        # )
        super(TransformerEncoder, self).__init__()
        print(f"Loading transformer model: {model_name}")
        self.transformer = SentenceTransformer(model_name)
        print(f"Selected formatter: {stack_formatter.name()}")
        self.stack_formatter = stack_formatter
        self.output_dim = out_dim
        self.coder = coder
        self.multi_stack = multi_stack
        self._name = model_name.split("/")[-2].strip()

    @lru_cache(maxsize=200_000)
    def forward(self, stack_id: int) -> torch.Tensor:
        frames = self.coder(stack_id, transformer=True)
        if self.multi_stack:
            frames = [self.stack_formatter.format(frame) for frame in frames]
        else:
            frames = self.stack_formatter.format(frames)
        emb = self.transformer.encode(frames, convert_to_tensor=True)

        return emb

    def opt_params(self) -> list:
        return list(self.transformer.parameters())

    def out_dim(self) -> int:
        return self.output_dim

    def name(self) -> str:
        return self._name

    # Make the method deprecated
    def format_stack(self, stack):
        warnings.warn(
            "The 'format_stack' method is deprecated and will be removed",
            DeprecationWarning,
            stacklevel=2,
        )
        # Select last 10 frames
        stack = list(dict.fromkeys(stack))
        stack = [frame for frame in stack if frame.lower() != "none"]
        return "\n".join([f"{i+1}: {frame}" for i, frame in enumerate(stack)])

    # @lru_cache(maxsize=200_000)
    def forward_all(self, stack_ids: list) -> torch.Tensor:
        # Process all stack_ids in batch
        frames_batch = [
            self.coder(stack_id, transformer=True) for stack_id in stack_ids
        ]

        # Join frames for each stack_id into a single string
        sentences = ["\n".join(frames) for frames in frames_batch]

        # Encode the entire batch of sentences at once
        embeddings = self.transformer.encode(sentences, convert_to_tensor=True)

        return embeddings
