import os
import time
import warnings
from abc import ABC, abstractmethod
from functools import lru_cache
from pprint import pprint
from timeit import timeit
from typing import List

import gensim
import numpy as np
import torch
from gensim.models import Word2Vec
from sentence_transformers import SentenceTransformer
from torch import nn

from data.formatters import StackFormatter
from data.stack_loader import StackLoader
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


class Frame2Vec:
    def __init__(self, model_path, vector_size=64, window=3, min_count=1, sg=1, negative=20, epochs=10):
        """
        Initialize the Frame2Vec model with Word2Vec parameters.
        """
        self.vector_size = vector_size
        self.model_path = model_path
        self.window = window
        self.min_count = min_count
        self.sg = sg  # Skip-gram
        self.negative = negative  # Negative sampling
        self.epochs = epochs
        self.model = None
    
    def tokenize_frame(self, frame: str) -> List[str]:
        parts = frame.split('.')
        return parts
    
    def train(self, stack_traces: List[List[str]]):
        """
        Train the Frame2Vec model on a list of stack traces. Load if exists, otherwise train and save.
        """
        if os.path.exists(self.model_path):
            print("Loading existing model from path", self.model_path)
            self.model = Word2Vec.load(self.model_path)
        else:
            print("Training new model...")
            tokenized_traces = [[self.tokenize_frame(frame) for frame in trace] for trace in stack_traces]
            tokenized_sentences = [item for sublist in tokenized_traces for item in sublist]

            self.model = Word2Vec(
                sentences=tokenized_sentences,
                vector_size=self.vector_size,
                window=self.window,
                min_count=self.min_count,
                sg=self.sg,
                workers=4,
                negative=self.negative,
                epochs=self.epochs,            
            )

            # Save the trained model
            self.model.save(self.model_path)
            print("Model saved successfully to path ", self.model_path)
    
    def get_vector(self, frame: str):
        """
        Get the vector representation of a stack frame.
        """
        sub_frames = self.tokenize_frame(frame)
        vectors = [self.model.wv[sub] for sub in sub_frames if sub in self.model.wv]

        # print(np.array(vectors).shape)
        
        return np.mean(vectors, axis=0)
    
    def encode(self, stack_trace: List[str]):
        """
        Convert an entire stack trace to its vectorized representation.
        """
        if len(stack_trace) == 0:
            stack_trace = ['unknown']
            
        frame_representations = [self.get_vector(frame) for frame in stack_trace]
        frame_representations = np.array(frame_representations)
        # Replace nan with zeros
        frame_representations = [
            rep if not np.isnan(rep).any() else np.zeros(self.vector_size) 
            for rep in frame_representations
        ]
        return np.array(frame_representations)  # torch.tensor(np.mean(frame_representations, axis=0))


class DeepCrashEncoder:
    def __init__(
        self,
        coder: SeqCoder,
        train_stack_ids: List[int],
        bucket_name: str,
        out_dim: int = 50,
        multi_stack: bool = False,
    ):
        super(DeepCrashEncoder, self).__init__()
        self.model_path = f"frame2vec_{bucket_name}.model"
        self.frame2vec = Frame2Vec(self.model_path, vector_size=out_dim)
        stacks = [coder(stack_id, transformer=True) for stack_id in train_stack_ids]
        self.frame2vec.train(stacks)
        print("Frame2Vec model trained successfully.")
        self.coder = coder
        self.output_dim = out_dim
        self._name = f"deepcrash_encoder_{bucket_name}"

    @lru_cache(maxsize=200_000)
    def forward(self, stack_id: int) -> torch.Tensor:
        frames = self.coder(stack_id, transformer=True)
        emb = self.frame2vec.encode(frames)
        emb = torch.tensor(emb, dtype=torch.float32).to(device)

        # Check if it matches the output dimension
        if emb.size()[-1] == 0:
            print(f"Invalid output dimension: {emb.size()}")

        return emb

    def opt_params(self) -> list:
        return []

    def out_dim(self) -> int:
        return self.output_dim

    def name(self) -> str:
        return self._name