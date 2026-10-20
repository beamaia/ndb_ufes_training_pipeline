"""Classification metrics shared by training and evaluation jobs."""

from __future__ import annotations

from typing import Iterable

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
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
    }


def class_key(name: str) -> str:
    """A class name as it appears in column and metric names: ``Leukoplakia with dysplasia`` becomes ``leukoplakia_with_dysplasia``."""
    return name.lower().replace(" ", "_").replace("-", "_")


def confusion_metrics(labels: Iterable[int], predictions: Iterable[int], class_names: list[str], prefix: str) -> dict[str, float]:
    """The confusion matrix as numbers: ``<prefix>_cm_<true class>_as_<predicted class>`` is how many of the true class were predicted as the other class."""
    matrix = confusion_matrix(np.asarray(list(labels), dtype=int), np.asarray(list(predictions), dtype=int), labels=list(range(len(class_names))))
    return {f"{prefix}_cm_{class_key(true)}_as_{class_key(predicted)}": float(matrix[i, j]) for i, true in enumerate(class_names) for j, predicted in enumerate(class_names)}


def roc_auc_metrics(labels: Iterable[int], probabilities: np.ndarray, class_names: list[str], prefix: str) -> dict[str, float]:
    """The area under the ROC curve of each class against the others, and their mean (macro).

    A class that is absent from ``labels`` (or the only one present) has no ROC curve and is left out of the mean.
    """
    y_true = np.asarray(list(labels), dtype=int)
    probabilities = np.asarray(probabilities, dtype=float)
    metrics = {}
    for index, name in enumerate(class_names):
        positive = y_true == index
        if positive.any() and not positive.all():
            metrics[f"{prefix}_roc_auc_{class_key(name)}"] = float(roc_auc_score(positive, probabilities[:, index]))
    if metrics:
        metrics[f"{prefix}_roc_auc_macro"] = float(np.mean(list(metrics.values())))
    return metrics
