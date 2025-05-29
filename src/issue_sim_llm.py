from time import time
from typing import Optional

import torch

from data.buckets.bucket_data import BucketData
from data.buckets.issues_data import BucketDataset
from evaluation.issue_sim import paper_metrics_iter
from methods.classic.hyperopt import PairStackBasedIssueHyperoptModel
from methods.neural.train_issue_sim_llm import train_issue_model
from methods.pair_stack_issue_model import (MaxIssueScorer,
                                            PairStackBasedSimModel)
from models_factory import create_classic_model, create_neural_model
from utils import random_seed, set_seed


def neural_issues(
    bucket_data: BucketData,
    max_len: Optional[int] = None,
    trim_len: int = 0,
    loss_name: str = "point",
    hyp_top_stacks: int = 20,
    hyp_top_issues: int = 5,
    epochs: int = 2,
    method_name: str = "",
    lang: str = "",
    multi_stack: bool = False,
    encoder_path: str = None,
    skip_training: bool = False,
    max_frames: int = -1,
):
    set_seed(random_seed)
    print("Dataset:", bucket_data.name)
    print("Method:", method_name)
    print("Trim: ", trim_len)
    print(
        "train_days",
        bucket_data.train_days,
        "test_days",
        bucket_data.test_days,
        "warmup_days",
        bucket_data.warmup_days,
        "val_days",
        bucket_data.val_days,
        "loss_name",
        loss_name,
    )
    bucket_data.load()
    stack_loader = bucket_data.stack_loader(multi_stack)
    dataset = BucketDataset(bucket_data)
    unsup_stacks = dataset.train_stacks()
    
    all_stack_ids = None
    if method_name == "llm":
        all_stack_ids = [event.st_id for event in bucket_data.actions]

    model = create_neural_model(
        stack_loader,
        unsup_stacks,
        max_len,
        trim_len,
        model_name=method_name,
        language=lang,
        multi_stack=multi_stack,
        bucket_name=bucket_data.name,
        max_frames=max_frames,
        encoder_path=encoder_path,
        all_stack_ids=all_stack_ids,
    )

    train_issue_model(
        model,
        dataset
    )

    # ps_model = PairStackBasedSimModel(model, MaxIssueScorer())

    # start = time()
    # new_preds = ps_model.predict(dataset.test())
    # print("Time to predict", time() - start)

    # draw_acc_at_th(new_preds, model_name=model.name())
    # paper_metrics_iter(new_preds)
