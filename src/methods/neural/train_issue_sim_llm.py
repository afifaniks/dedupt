import copy
import math
import os
import random
import sys
from itertools import islice
from time import time
from typing import List, Tuple

import numpy as np
import torch
from sklearn.metrics import roc_auc_score, roc_curve
from tqdm import tqdm

from data.buckets.bucket_data import BucketData
from data.buckets.issues_data import BucketDataset, StackAdditionState
from data.pair_sim_selector import RandomPairSimSelector
from data.triplet_selector import RandomTripletSelector
from evaluation.issue_sim import paper_metrics_iter, score_model
from methods.neural.losses import (LossComputer, PointLossComputer,
                                   RanknetLossComputer, TripletLossComputer)
from methods.neural.neural_base import NeuralModel
from methods.pair_stack_issue_model import (MaxIssueScorer,
                                            PairStackBasedSimModelRAG)


def log_metrics(
    sim_stack_model: NeuralModel,
    loss_computer: LossComputer,
    train_sim_pairs_data_for_score: List[Tuple[int, int, int]],
    test_sim_pairs_data_for_score: List[Tuple[int, int, int]],
    train_data_for_score: List[StackAdditionState],
    test_data_for_score: List[StackAdditionState],
    prefix: str,
    writer,
    n_iter: int,
):
    sim_stack_model.eval()
    with torch.no_grad():
        train_loss_value = loss_computer.get_eval_raws(train_sim_pairs_data_for_score)
        test_loss_value = loss_computer.get_eval_raws(test_sim_pairs_data_for_score)

        if "trainable_encoder" in sim_stack_model.encoder.name():
            sim_stack_model.encoder.enable_cache()

        ps_model = PairStackBasedSimModelRAG(sim_stack_model, MaxIssueScorer())
        train_preds = ps_model.predict(train_data_for_score)
        test_preds = ps_model.predict(test_data_for_score)
        train_score = score_model(train_preds, full=False)
        test_score = score_model(test_preds, full=False)

        if "trainable_encoder" in sim_stack_model.encoder.name():
            sim_stack_model.encoder.disable_cache()
    print(
        prefix + f"Train loss: {round(train_loss_value, 4)}. "
        f"Test loss: {round(test_loss_value, 4)}. "
        f"Train prec {train_score[0]}, rec {train_score[1]}. "
        f"Test prec {test_score[0]}, rec {test_score[1]}       ",
        end="",
    )

    if writer:
        writer.add_scalar("Loss/train", train_loss_value, n_iter)
        writer.add_scalar("Loss/test", test_loss_value, n_iter)

    return train_loss_value, test_loss_value, train_score, test_score


def log_all_data_scores(sim_stack_model: NeuralModel, data_gen):
    sim_stack_model.eval()
    data_gen.reset()
    ps_model = PairStackBasedSimModelRAG(
        sim_stack_model, MaxIssueScorer()
    )  # =None for no filter

    # train_preds = ps_model.predict(data_gen.train())
    # print("Train")
    # # score_model(train_preds, th=-1, model_name="Train " + sim_stack_model.name())
    # paper_metrics_iter(train_preds)

    # test_preds = ps_model.predict(data_gen.test())
    test_preds = ps_model.predict(data_gen.test())
    print("Test")
    # te_score = score_model(test_preds, model_name="Test " + sim_stack_model.name())
    paper_metrics_iter(test_preds)

    print()


def train_issue_model(
    sim_stack_model: NeuralModel,
    data_gen: BucketDataset,
):

    # To enable auc calculation comment out the following line and uncomment the next line
    log_all_data_scores(sim_stack_model, data_gen)
    # log_metrics_auc(sim_stack_model, data_gen, bucket_data)
