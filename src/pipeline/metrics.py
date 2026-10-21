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


TRAINING_METRICS = {"learning_rate", "epoch_seconds", "epochs_run", "train_seconds", "peak_gpu_memory_mb", "early_stopped_epoch", "fold_seconds", "checkpoint_mb"}
RUN_METRICS = {"run_seconds", "cost_usd"}
TEST_UNITS = ("picture_vs_ndb", "picture", "patient")  # what the test metrics can be about besides the patches; picture_vs_ndb has to come before picture


def board_name(key: str) -> str:
    """The name of a metric on the MLflow board. A slash makes a section there, so the metrics are grouped instead of one list of about 90:
    ``val_balanced_acc`` is ``val/balanced_acc``, ``heldout_test_roc_auc_oscc`` is ``test_patch/roc_auc_oscc``, ``heldout_test_picture_cm_oscc_as_oscc``
    is ``test_picture_cm/oscc_as_oscc`` (``test_patch``, ``test_picture``, ``test_patient`` and ``test_picture_vs_ndb`` each have a section, and each ``_cm``
    section holds the cells of a confusion matrix). The time, memory and size metrics are in ``training/`` and ``run/``."""
    if key in TRAINING_METRICS:
        return f"training/{key}"
    if key in RUN_METRICS:
        return f"run/{key}"
    stage, _, rest = key.removeprefix("heldout_").partition("_")
    group = stage
    if stage == "test":
        group = "test_patch"
        for unit in TEST_UNITS:
            if rest.startswith(f"{unit}_"):
                group, rest = f"test_{unit}", rest[len(unit) + 1:]
                break
    if rest.startswith("cm_"):
        group, rest = f"{group}_cm", rest[3:]
    return f"{group}/{rest}" if rest else key


def board(metrics: dict[str, float], suffix: str = "") -> dict[str, float]:
    """The metrics under their board names (see ``board_name``), with ``suffix`` added to each name."""
    return {board_name(key) + suffix: value for key, value in metrics.items()}
