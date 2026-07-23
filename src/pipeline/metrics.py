"""Classification metrics shared by training and evaluation jobs."""

from __future__ import annotations

from typing import Iterable

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)


def classification_metrics(
    labels: Iterable[int],
    predictions: Iterable[int],
    num_classes: int,
    prefix: str,
) -> dict[str, float]:
    """Return aggregate metrics with explicit macro and micro definitions."""

    y_true = np.asarray(list(labels), dtype=int)
    y_pred = np.asarray(list(predictions), dtype=int)
    if y_true.size == 0 or y_pred.size == 0:
        raise ValueError("Cannot compute classification metrics for an empty evaluation split.")
    if y_true.shape != y_pred.shape:
        raise ValueError("Labels and predictions must have the same length.")
    class_labels = list(range(num_classes))

    return {
        f"{prefix}_acc": float(accuracy_score(y_true, y_pred)),
        f"{prefix}_balanced_acc": float(
            balanced_accuracy_score(y_true, y_pred)
        ),
        f"{prefix}_precision": float(
            precision_score(
                y_true,
                y_pred,
                labels=class_labels,
                average="macro",
                zero_division=0,
            )
        ),
        f"{prefix}_recall": float(
            recall_score(
                y_true,
                y_pred,
                labels=class_labels,
                average="macro",
                zero_division=0,
            )
        ),
        f"{prefix}_macro_f1": float(
            f1_score(
                y_true,
                y_pred,
                labels=class_labels,
                average="macro",
                zero_division=0,
            )
        ),
        f"{prefix}_micro_precision": float(
            precision_score(y_true, y_pred, average="micro", zero_division=0)
        ),
        f"{prefix}_micro_recall": float(
            recall_score(y_true, y_pred, average="micro", zero_division=0)
        ),
    }
