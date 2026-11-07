import argparse
import os
import re
from pathlib import Path

import mlflow
import pandas as pd

from src.config import load_env

REPO = Path(__file__).resolve().parents[1]
CLASSES = ["leukoplakia_without_dysplasia", "leukoplakia_with_dysplasia", "oscc"]  # from the least to the most severe
EXPERIMENTS = {"experiment_1": "ndb_ufes_exp1", "experiment_2": "ndb_ufes_exp2"}  # the folders of the export and the MLflow experiments they come from
CELL = re.compile(r"metrics\.(test_(?:patch|picture|patient)_cm)/(?P<true>.+)_as_(?P<predicted>.+)")
BALANCED = {"patch": "test_patch/balanced_acc", "picture": "test_picture/balanced_acc", "patient": "test_patient/balanced_acc"}


def mean_std(values: pd.Series) -> str:
    return f"{values.mean() * 100:.1f} ± {values.std(ddof=0) * 100:.1f}"


def recalls(folds: pd.DataFrame) -> dict[str, str]:
    """The recall of each class over all the fold runs of a model: its correct patches over its patches, from the confusion cells added up."""
    cell = lambda true, predicted: folds[f"metrics.test_patch_cm/{true}_as_{predicted}"].sum()
    return {f"recall {name}": f"{cell(name, name) / sum(cell(name, predicted) for predicted in CLASSES) * 100:.1f}%" for name in CLASSES}


def summarize(runs: pd.DataFrame) -> pd.DataFrame:
    """One row for each model, from the runs of an experiment (``mlflow.search_runs``): the fold runs give the metrics, the parent run gives the cost."""
    is_fold = runs["tags.mlflow.parentRunId"].notna()
    folds, parents = runs[is_fold], runs[~is_fold].set_index("tags.model")
    rows = []
    for model, group in folds.groupby("tags.model"):
        row = {"model": model, "folds": len(group)}
        row |= {f"balanced acc {label}": mean_std(group[f"metrics.{metric}"]) for label, metric in BALANCED.items()}
        row["ROC AUC"] = mean_std(group["metrics.test_patch/roc_auc_macro"])
        row |= {f"AUC {name}": mean_std(group[f"metrics.test_patch/roc_auc_{name}"]) for name in CLASSES}  # each class against the other two
        row |= recalls(group)
        row["epochs"] = f"{group['metrics.training/epochs_run'].mean():.0f}"
        row["minutes"] = f"{group['metrics.training/fold_seconds'].sum() / 60:.0f}"
        cost = parents["metrics.run/cost_usd"].get(model) if "metrics.run/cost_usd" in parents else None
        row["cost $"] = "" if pd.isna(cost) else f"{cost:.2f}"
        rows.append(row | {"_sort": group["metrics.test_patch/balanced_acc"].mean()})
    return pd.DataFrame(rows).sort_values("_sort", ascending=False).drop(columns="_sort").reset_index(drop=True)


def best_epoch_validation(runs: pd.DataFrame) -> pd.DataFrame:
    """The balanced accuracy on the validation fold at the epoch whose checkpoint is kept (the lowest validation loss), for each model and fold: the parent
    run of a model logs it once for each fold, with the fold as the step."""
    client = mlflow.MlflowClient()
    parents = runs[runs["tags.mlflow.parentRunId"].isna()]
    rows = [{"model": parent["tags.model"], "fold": str(metric.step), "val_balanced_acc_at_best_epoch": metric.value}
            for _, parent in parents.iterrows() for metric in client.get_metric_history(parent["run_id"], "val/balanced_acc")]
    return pd.DataFrame(rows)


def fold_metrics(runs: pd.DataFrame) -> pd.DataFrame:
    """One row for each fold run of each model, with every test and training metric (the confusion cells have their own table) and the validation score
    of the kept checkpoint, which is what picks a representative fold of a model without looking at the test fold."""
    folds = runs[runs["tags.mlflow.parentRunId"].notna()]
    keep = [column for column in folds.columns if column.startswith("metrics.") and not CELL.match(column) and not column.startswith("metrics.val")]
    table = folds[["run_id", "tags.model", "tags.validation_fold", *keep]].rename(columns={"tags.model": "model", "tags.validation_fold": "fold"})
    table.columns = [column.removeprefix("metrics.") for column in table.columns]
    table = table.merge(best_epoch_validation(runs), on=["model", "fold"], how="left", validate="one_to_one")
    return table.sort_values(["model", "fold"]).reset_index(drop=True)


def confusion_matrices(runs: pd.DataFrame) -> pd.DataFrame:
    """The confusion cells of each model and level (patch, picture, patient), added up over its fold runs, with the share of the true class."""
    folds = runs[runs["tags.mlflow.parentRunId"].notna()]
    rows = []
    for column in folds.columns:
        cell = CELL.match(column)
        if cell:
            for model, count in folds.groupby("tags.model")[column].sum().items():
                rows.append({"model": model, "level": cell[1].removesuffix("_cm").removeprefix("test_"), "true": cell["true"], "predicted": cell["predicted"], "count": int(count)})
    table = pd.DataFrame(rows)
    table["share of true class %"] = (table["count"] / table.groupby(["model", "level", "true"])["count"].transform("sum") * 100).round(1)
    return table.sort_values(["model", "level", "true", "predicted"]).reset_index(drop=True)


def export(runs: pd.DataFrame, folder: Path) -> None:
    folder.mkdir(parents=True, exist_ok=True)
    table = summarize(runs)
    table.to_csv(folder / "summary.csv", index=False)
    (folder / "summary.md").write_text(markdown(table) + "\n")
    fold_metrics(runs).to_csv(folder / "fold_metrics.csv", index=False)
    confusion_matrices(runs).to_csv(folder / "confusion_matrices.csv", index=False)


def finished_runs(experiment: str) -> pd.DataFrame:
    runs = mlflow.search_runs(experiment_names=[experiment], filter_string="attributes.status = 'FINISHED'")
    if runs.empty:
        raise SystemExit(f"No finished runs in the experiment {experiment}")
    return runs


def markdown(table: pd.DataFrame) -> str:
    lines = ["| " + " | ".join(table.columns) + " |", "| " + " | ".join("---" for _ in table.columns) + " |"]
    lines += ["| " + " | ".join(str(value) for value in row) + " |" for row in table.itertuples(index=False)]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--experiment", help="The MLflow experiment, for example ndb_ufes_exp2.")
    parser.add_argument("--csv", type=Path, help="Also write the table to this CSV file.")
    parser.add_argument("--export", type=Path, help="Write the tables of both experiments (summary, fold metrics, confusion matrices) to a folder for each, for example ../ndb_ufes_data_organizer/results/experiments.")
    args = parser.parse_args()
    load_env(REPO / ".env", Path(os.getenv("MLFLOW_ENV_FILE", REPO.parent / "ndb_ufes_mlflow" / ".env")))
    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://localhost:8000/"))
    if args.export:
        for folder, experiment in EXPERIMENTS.items():
            export(finished_runs(experiment), args.export / folder)
            print(f"{experiment} -> {args.export / folder}")
        return
    if not args.experiment:
        parser.error("give --experiment or --export")
    table = summarize(finished_runs(args.experiment))
    print(markdown(table))
    if args.csv:
        table.to_csv(args.csv, index=False)


if __name__ == "__main__":
    main()
