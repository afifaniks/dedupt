from typing import List

from data.formatters import JavaStackFormatter, get_formatter
from data.stack_loader import StackLoader
from methods.base import SimStackModel
from methods.classic.brodie import BrodieModel
from methods.classic.cosine import CosineModel
from methods.classic.durfex import DurfexModel
from methods.classic.lerch import LerchModel
from methods.classic.levenshtein import LevenshteinModel
from methods.classic.moroo import MorooModel
from methods.classic.prefix_match import PrefixMatchModel
from methods.classic.rebucket import RebucketModel
from methods.classic.trace_sim import TraceSimModel
from methods.neural import device
from methods.neural.neural_base import NeuralModel
from methods.neural.siam.aggregation import ConcatAggregation
from methods.neural.siam.encoders import LSTMEncoder, TransformerEncoder
from methods.neural.siam.siam_network import (
    SiamMultiModalModel,
    SiamSentTransformerModel,
    SiamSentTransformerModelMultiStack,
    SiamSentTransformerModelMultiStackAttention,
    SiamSentTransformerModelMultiStackMultiHead,
)
from preprocess.entry_coders import Stack2Seq, Stack2SeqMultiStack
from preprocess.seq_coder import SeqCoder, SeqCoderMulti
from preprocess.tokenizers import SimpleTokenizer


def create_classic_model(
    stack_loader: StackLoader,
    method: str = "lerch",
    max_len: int = None,
    trim_len: int = 0,
    sep: str = ".",
) -> SimStackModel:
    stack2seq = Stack2Seq(cased=False, trim_len=trim_len, sep=sep)
    coder = SeqCoder(
        stack_loader, stack2seq, SimpleTokenizer(), min_freq=0, max_len=max_len
    )
    if method == "lerch":
        model = LerchModel(coder)
    elif method == "cosine":
        model = CosineModel(coder)
    elif method == "prefix":
        model = PrefixMatchModel(coder)
    elif method == "rebucket":
        model = RebucketModel(coder)
    elif method == "tracesim":
        model = TraceSimModel(coder)
    elif method == "levenshtein":
        model = LevenshteinModel(coder)
    elif method == "brodie":
        model = BrodieModel(coder)
    elif method == "moroo":
        model = MorooModel(coder)
    elif method == "durfex":
        model = DurfexModel(coder, ns=(1, 2, 3))
    else:
        raise ValueError("Method name is not match")
    return model


def create_neural_model(
    stack_loader: StackLoader,
    unsup_data: List[int],
    max_len: int = None,
    trim_len: int = 0,
    sep: str = ".",
    model_name: str = "s3m",
    language: str = "",
    multi_stack: bool = False,
    bucket_name: str = "",
    max_frames: int = 10,
) -> NeuralModel:
    stack2seq = Stack2Seq(cased=False, trim_len=trim_len, sep=sep)
    coder = SeqCoder(
        stack_loader, stack2seq, SimpleTokenizer(), min_freq=0, max_len=max_len
    )
    model = None

    if model_name == "transformer":
        print("Multi stack status:", multi_stack)
        stack_formatter = get_formatter(language, max_frames)
        if multi_stack:
            stack2seq = Stack2SeqMultiStack(cased=False, trim_len=trim_len, sep=sep)
            coder = SeqCoderMulti(
                stack_loader, stack2seq, SimpleTokenizer(), min_freq=0, max_len=max_len
            )

        encoder = TransformerEncoder(
            coder=coder,
            stack_formatter=stack_formatter,
            model_name=f"models/bge-base-{bucket_name}_10_class/final",
            multi_stack=multi_stack,
        )

        if multi_stack:
            model = SiamSentTransformerModelMultiStack(
                encoder, features_num=4, out_num=1
            )
        else:
            model = SiamSentTransformerModel(encoder, features_num=4, out_num=1)
    elif model_name == "s3m":
        coder.fit(unsup_data)
        encoders = [LSTMEncoder(coder, dim=50, hid_dim=100).to(device)]
        model = SiamMultiModalModel(
            encoders, ConcatAggregation, features_num=4, out_num=1
        ).to(device)
    else:
        raise ValueError("Model name does not match")

    model.to(device)
    print("Choosen model:", model)

    return model
