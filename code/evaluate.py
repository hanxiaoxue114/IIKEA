"""
Evaluation metrics for II-KEA diagnosis prediction.

Metrics (as used in the paper):
  - w-F1  : weighted F1 score across all disease labels
  - R@k   : Recall at k — proportion of true-positive codes in top-k predictions
             relative to total number of positive codes
"""

from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
from sklearn.metrics import f1_score
from sklearn.preprocessing import MultiLabelBinarizer


def recall_at_k(
    y_true: List[List[str]],
    y_pred_ranked: List[List[str]],
    k: int,
) -> float:
    """
    R@k = (1/N) * sum_p  |top_k(pred_p) ∩ true_p| / |true_p|

    Args:
        y_true:        list of ground-truth code lists per patient
        y_pred_ranked: list of predicted code lists per patient,
                       ordered from most to least confident
        k:             cutoff

    Returns:
        Mean recall@k across patients (patients with empty ground truth excluded).
    """
    scores = []
    for true_codes, pred_codes in zip(y_true, y_pred_ranked):
        if not true_codes:
            continue
        top_k = set(pred_codes[:k])
        true_set = set(true_codes)
        scores.append(len(top_k & true_set) / len(true_set))
    return float(np.mean(scores)) if scores else 0.0


def weighted_f1(
    y_true: List[List[str]],
    y_pred: List[List[str]],
    all_labels: List[str],
) -> float:
    """
    w-F1: weighted F1 score (weighted by support per class).

    Args:
        y_true:     ground-truth multi-label lists
        y_pred:     predicted multi-label lists
        all_labels: complete vocabulary of possible labels
    """
    mlb = MultiLabelBinarizer(classes=all_labels)
    Y_true = mlb.fit_transform(y_true)
    Y_pred = mlb.transform(y_pred)
    return float(f1_score(Y_true, Y_pred, average="weighted", zero_division=0))


def evaluate(
    y_true: List[List[str]],
    y_pred_ranked: List[List[str]],
    all_labels: List[str],
    k_values: List[int] = (10, 20),
) -> Dict[str, float]:
    """
    Compute all metrics and return as a dict.

    Args:
        y_true:         ground-truth code lists per patient
        y_pred_ranked:  predicted codes per patient, ranked by confidence
        all_labels:     full disease vocabulary
        k_values:       list of k for R@k

    Returns:
        e.g. {"w-F1": 28.61, "R@10": 38.52, "R@20": 43.86}
    """
    results: Dict[str, float] = {}

    # w-F1 uses the full predicted set (not truncated)
    results["w-F1"] = weighted_f1(y_true, y_pred_ranked, all_labels) * 100

    for k in k_values:
        results[f"R@{k}"] = recall_at_k(y_true, y_pred_ranked, k) * 100

    return results


def print_results(results: Dict[str, float]) -> None:
    print("\n── Evaluation Results ──────────────────")
    for metric, value in results.items():
        print(f"  {metric:<8} {value:.2f}")
    print("────────────────────────────────────────\n")
