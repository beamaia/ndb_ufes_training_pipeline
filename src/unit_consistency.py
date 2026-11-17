"""Are the errors of the fold runs of a model on the same patches, pictures and patients?

Each model is trained once for each validation fold (five versions) and all versions are tested on the same test fold. For every test patch, picture
and patient, this counts in how many versions of a model the call is wrong. The picture and patient calls are made as in training (the mean probabilities of the patches,
``results_summary.summarize_predictions``). If the errors were independent between versions, some units would be wrong in all versions only by chance; if
a unit is hard for the architecture, many more are wrong in all versions than chance gives.

The tables are written to ``--out``:
- ``unit_consistency.csv``: for each model and unit type, how many units are always right, always wrong and mixed, and how many are always wrong by chance.
- ``unit_wrong_versions.csv``: for each model and unit, in how many versions it is wrong.
"""

import argparse
import os
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd

import src.pipeline  # noqa: F401  results_summary needs src.pipeline.metrics, and the package imports results_summary back: load the package first
from src import logger
from src.config import load_env
from src.panel import REPO, load_predictions, prepare
from src.results_summary import summarize_predictions

CLASS_NAMES = ["Leukoplakia without dysplasia", "Leukoplakia with dysplasia", "OSCC"]  # the class numbers 0, 1, 2
UNITS = {"patch": "patch_id", "picture": "histopath_id", "patient": "case_group"}  # the column that says which unit a patch belongs to


def right_by_version(runs: pd.DataFrame, unit: str) -> pd.DataFrame:
    """The test calls of one model, one row for each unit and one column for each version (fold run): is the call right?

    For patches, the call is the class of the patch; for pictures and patients, the class of the mean probabilities of their patches, against the label that most of
    their patches have.
    """
    column, versions = UNITS[unit], {}
    for run_id, group in runs.groupby("run_id"):
        if unit == "patch":
            versions[run_id] = pd.Series((group["y_pred"] == group["y_true"]).to_numpy(), index=group["patch_id"].to_numpy())
        else:
            table = summarize_predictions(group, CLASS_NAMES, column).set_index(column)
            versions[run_id] = table["y_pred"] == table["y_true"]
    return pd.DataFrame(versions)


def consistency(right: pd.DataFrame) -> dict[str, float]:
    """How the units split by the number of versions that get them right, and how many would be wrong in all versions if the versions erred independently.

    ``chance_always_wrong`` is the number of units times the product of the error rates of the versions.
    """
    n_right = right.sum(axis=1)
    versions = right.shape[1]
    return {
        "units": len(right), "versions": versions, "mean_accuracy": float(right.to_numpy().mean()),
        "always_right": int((n_right == versions).sum()), "always_wrong": int((n_right == 0).sum()), "mixed": int(((n_right > 0) & (n_right < versions)).sum()),
        "chance_always_wrong": float(len(right) * np.prod(1 - right.mean(axis=0).to_numpy())),
    }


def run(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """The summary of each model and unit type, and the number of versions in which each unit is wrong."""
    summary, wrong = [], []
    for model, runs in frame.groupby("model"):
        for unit in UNITS:
            right = right_by_version(runs, unit)
            summary.append({"model": model, "unit": unit} | consistency(right))
            wrong.append(pd.DataFrame({"model": model, "unit": unit, "unit_id": right.index, "wrong_versions": (~right).sum(axis=1).to_numpy()}))
    return pd.DataFrame(summary), pd.concat(wrong, ignore_index=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--experiment", default="ndb_ufes_exp2", help="The MLflow experiment, for example ndb_ufes_exp2.")
    parser.add_argument("--out", type=Path, required=True, help="The folder for the tables, for example ../ndb_ufes_data_organizer/results/experiments/experiment_2.")
    args = parser.parse_args()
    os.environ.setdefault("MLFLOW_ENABLE_ARTIFACTS_PROGRESS_BAR", "false")
    load_env(REPO / ".env", Path(os.getenv("MLFLOW_ENV_FILE", REPO.parent / "ndb_ufes_mlflow" / ".env")))
    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://localhost:8000/"))
    summary, wrong = run(prepare(load_predictions(args.experiment)))
    args.out.mkdir(parents=True, exist_ok=True)
    summary.round(3).to_csv(args.out / "unit_consistency.csv", index=False)
    wrong.to_csv(args.out / "unit_wrong_versions.csv", index=False)
    logger.info(f"Wrote unit_consistency.csv and unit_wrong_versions.csv to {args.out}")
    print(summary.groupby("unit")[["units", "always_right", "always_wrong", "mixed", "chance_always_wrong"]].mean().round(1).reindex(list(UNITS)).to_string())


if __name__ == "__main__":
    main()
