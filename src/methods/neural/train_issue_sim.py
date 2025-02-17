import copy
import math
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
from methods.neural.losses import (
    LossComputer,
    PointLossComputer,
    RanknetLossComputer,
    TripletLossComputer,
)
from methods.neural.neural_base import NeuralModel
from methods.pair_stack_issue_model import MaxIssueScorer, PairStackBasedSimModel


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

        ps_model = PairStackBasedSimModel(sim_stack_model, MaxIssueScorer())
        train_preds = ps_model.predict(train_data_for_score)
        test_preds = ps_model.predict(test_data_for_score)
        train_score = score_model(train_preds, full=False)
        test_score = score_model(test_preds, full=False)
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


def log_metrics_auc(
    sim_stack_model: NeuralModel,
    test_data_for_score: List[StackAdditionState],
    bucket_data: BucketData,
):
    sim_stack_model.eval()
    test_data_for_score.reset()
    test_data_for_score = test_data_for_score.test_auc()

    all_events = []
    for i, event in tqdm(enumerate(test_data_for_score)):
        all_events.append(event)

    # Choose 100 random events where 50 event.is_id == event.st_id and 50 event.is_id != event.st_id
    same_id_events = [event for event in all_events if event.is_id == event.st_id]
    diff_id_events = [event for event in all_events if event.is_id != event.st_id]

    print(
        "Same ID events:", len(same_id_events), "Diff ID events:", len(diff_id_events)
    )

    if len(same_id_events) < 50 or len(diff_id_events) < 50:
        raise ValueError("Not enough events to sample from")

    total_auc = 0
    num_iter = 50

    for iter_ in range(num_iter):
        selected_same_id_events = random.sample(same_id_events, 50)
        selected_diff_id_events = random.sample(diff_id_events, 50)

        selected_events = selected_same_id_events + selected_diff_id_events
        random.shuffle(selected_events)

        with torch.no_grad():
            ps_model = PairStackBasedSimModel(sim_stack_model, MaxIssueScorer())
            test_preds = ps_model.predict(selected_events)
            # test_score = score_model(test_preds, full=False)

            # Calculate auc metric
            y_true, y_pred = [], []
            for issue_id, actual_dup_id, prs in test_preds:
                sorted_pr = sorted(prs.items(), key=lambda x: -x[1])
                y_true.append(0 if bucket_data.get_dup_id(issue_id) == issue_id else 1)
                y_pred.append(sorted_pr[0][1])

            y_true = np.array(y_true)
            y_pred = np.array(y_pred)

            auc = roc_auc_score(y_true, y_pred)
            fpr, tpr, th = roc_curve(y_true, y_pred, pos_label=1)
            total_auc += auc
            print(f"Sample {iter_}: AUC: {auc}, \nFPR: {fpr}, \nTPR: {tpr} \nTH: {th}")

    print("Mean AUC:", total_auc / num_iter)
    exit()


def log_all_data_scores(sim_stack_model: NeuralModel, data_gen):
    sim_stack_model.eval()
    data_gen.reset()
    ps_model = PairStackBasedSimModel(
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
    bucket_data: BucketData,
    loss_name: str,
    optimizers: List,
    epochs: int = 1,
    batch_size: int = 25,
    selection_from_event_num: int = 4,
    writer=None,
    period: int = 25,
    skip_training=False,
):
    if loss_name == "point":
        train_selector = RandomPairSimSelector(selection_from_event_num)
        loss_computer = PointLossComputer(sim_stack_model, train_selector)
    elif loss_name == "ranknet":
        train_selector = RandomTripletSelector(selection_from_event_num)
        # test_selector = RandomTripletSelector(2)
        loss_computer = RanknetLossComputer(sim_stack_model, train_selector)
    elif loss_name == "triplet":
        train_selector = RandomTripletSelector(selection_from_event_num)
        # test_selector = RandomTripletSelector(2)
        loss_computer = TripletLossComputer(sim_stack_model, train_selector, margin=0.2)
    else:
        raise ValueError

    train_data_for_score = [copy.deepcopy(x) for x in islice(data_gen.train(), 50)]
    test_data_for_score = [copy.deepcopy(x) for x in islice(data_gen.test(), 50)]
    train_sim_pairs_data_for_score = list(train_selector.generate(train_data_for_score))
    test_sim_pairs_data_for_score = list(train_selector.generate(test_data_for_score))
    print("Data sample:", train_data_for_score[0])
    data_gen.reset()
    assert len(train_sim_pairs_data_for_score) > 0

    start = time()
    print("Time to score validation data:", time() - start)

    n_iter = 0
    best_loss = math.inf
    print("Total epochs:", epochs)
    if skip_training:
        print("Skipping training")
    else:
        for epoch in range(epochs):
            data_gen.reset()
            for i, event in tqdm(
                enumerate(data_gen.train()),
                desc="Train Step",
                file=sys.stderr,
                dynamic_ncols=False,
                ascii=True,
            ):
                # log_all_data_scores(sim_stack_model, data_gen)
                sim_stack_model.train(True)
                loss = loss_computer.get_event(event)
                if loss is None:
                    continue
                loss.backward()

                if i != 0 and i % batch_size == 0:
                    for optimizer in optimizers:
                        optimizer.step()
                        optimizer.zero_grad()
                # sim_stack_model.train(False)
                # if i == 0 or (i + 1) % period == 0:
                #     prefix = f"\rEpoch {epoch}: {i + 1}. "
                #     log_metrics(
                #         sim_stack_model,
                #         loss_computer,
                #         train_sim_pairs_data_for_score,
                #         test_sim_pairs_data_for_score,
                #         train_data_for_score,
                #         test_data_for_score,
                #         prefix,
                #         writer,
                #         n_iter,
                #     )
                # if (i + 1) % 1000 == 0:
                #     print()
                n_iter += 1
            sim_stack_model.train(False)
            prefix = f"\rEpoch {epoch}: {i + 1}. "
            _, test_loss, _, _ = log_metrics(
                sim_stack_model,
                loss_computer,
                train_sim_pairs_data_for_score,
                test_sim_pairs_data_for_score,
                train_data_for_score,
                test_data_for_score,
                prefix,
                writer,
                n_iter,
            )
            print()
            print(f"Epoch {epoch} done.")

            if test_loss < best_loss:
                print("Loss improved. Saving new best model...")
                best_loss = test_loss
                torch.save(
                    sim_stack_model.state_dict(),
                    f"/home/mdafifal.mamun/research/S3M/models/{sim_stack_model.name()}.pth",
                )

    # Load best model
    print("Loading best model...")
    sim_stack_model.load_state_dict(
        torch.load(
            f"/home/mdafifal.mamun/research/S3M/models/{sim_stack_model.name()}.pth",
            weights_only=True,
        )
    )
    # log_metrics_auc(sim_stack_model, data_gen, bucket_data)
    log_all_data_scores(sim_stack_model, data_gen)

    if writer:
        writer.close()
    return sim_stack_model
