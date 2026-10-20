import numpy as np
import pandas as pd

from src.pipeline.metrics import class_key, classification_metrics, confusion_metrics, roc_auc_metrics

UNITS = {"picture": "histopath_id", "patient": "case_group"}


def summarize_predictions(predictions: pd.DataFrame, class_names: list[str], unit_column: str) -> pd.DataFrame:
    """One row per picture (or patient): the mean class probabilities of its patches, the predicted class, and the class most of its patches have.

    A tie between classes goes to the more severe one (the later class in ``class_names``).
    """
    probability_columns = [f"prob_{class_key(name)}" for name in class_names]
    rows = []
    for unit, group in predictions.groupby(unit_column):
        counts = np.bincount(group["y_true"].astype(int), minlength=len(class_names))
        mean_probabilities = group[probability_columns].mean().to_numpy()
        rows.append({
            unit_column: unit,
            "patches": len(group),
            "y_true": int(len(counts) - 1 - np.argmax(counts[::-1])),
            "y_pred": int(np.argmax(mean_probabilities)),
            **dict(zip(probability_columns, mean_probabilities)),
        })
    return pd.DataFrame(rows)


def summarize_metrics(predictions: pd.DataFrame, class_names: list[str], prefix: str) -> tuple[dict[str, float], dict[str, pd.DataFrame]]:
    """The metrics of the per-picture and per-patient predictions, under ``<prefix>_picture_*`` and ``<prefix>_patient_*``, and the tables behind them.

    The patch table must have the columns ``histopath_id`` and ``case_group``. For the pictures that are in NDB-UFES, the
    prediction is also compared with the diagnosis of the picture (``<prefix>_picture_vs_ndb_*``).
    """
    metrics, tables = {}, {}
    for unit, column in UNITS.items():
        table = summarize_predictions(predictions, class_names, column)
        tables[unit] = table
        probabilities = table[[f"prob_{class_key(name)}" for name in class_names]].to_numpy()
        metrics |= classification_metrics(table["y_true"], table["y_pred"], len(class_names), prefix=f"{prefix}_{unit}")
        metrics |= roc_auc_metrics(table["y_true"], probabilities, class_names, prefix=f"{prefix}_{unit}")
        metrics |= confusion_metrics(table["y_true"], table["y_pred"], class_names, prefix=f"{prefix}_{unit}")
        metrics[f"{prefix}_{unit}_count"] = float(len(table))
    if "ndb_diagnosis" in predictions:
        known = predictions[predictions["ndb_diagnosis"] != ""]
        if len(known):
            pictures = summarize_predictions(known, class_names, "histopath_id")
            diagnosis = known.groupby("histopath_id")["ndb_diagnosis"].first().map({name: index for index, name in enumerate(class_names)})
            pictures["y_true"] = pictures["histopath_id"].map(diagnosis)
            metrics |= classification_metrics(pictures["y_true"], pictures["y_pred"], len(class_names), prefix=f"{prefix}_picture_vs_ndb")
            metrics |= confusion_metrics(pictures["y_true"], pictures["y_pred"], class_names, prefix=f"{prefix}_picture_vs_ndb")
            metrics[f"{prefix}_picture_vs_ndb_count"] = float(len(pictures))
    return metrics, tables
