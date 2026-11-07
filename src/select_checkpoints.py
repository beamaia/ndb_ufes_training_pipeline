import argparse
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[1]
MLFLOW = REPO.parent / "ndb_ufes_mlflow"
BUCKET = "ndb-ufes-mlflow"  # the bucket of ndb_ufes_mlflow/scripts/archive_checkpoints.sh
MODELS = ["vgg16", "densenet121", "swin_tiny_patch4_window7_224"]  # the best CNN, the CNN of the article, and the best transformer (see model_selection.md)
SCORE = "val_balanced_acc_at_best_epoch"


def representative(folds: pd.DataFrame, manifest: pd.DataFrame, models: list[str]) -> pd.DataFrame:
    """For each model, the fold run with the highest balanced accuracy on its validation fold, at the epoch of the checkpoint that was kept. The test fold
    plays no part, so the choice does not look at the data the maps will be drawn on. The checkpoint is its path in ``var/mlflow/artifacts`` and in S3."""
    chosen = folds[folds["model"].isin(models)].sort_values(SCORE, ascending=False).drop_duplicates("model")
    files = manifest.assign(run_id=manifest["path"].str.split("/").str[1]).rename(columns={"path": "checkpoint"})[["run_id", "checkpoint", "bytes"]]
    chosen = chosen.merge(files, on="run_id", how="left", validate="one_to_one")
    chosen["megabytes"] = (chosen["bytes"] / 2**20).round(0)
    return chosen[["model", "fold", "run_id", SCORE, "checkpoint", "megabytes"]].round(4).reset_index(drop=True)


def restore_commands(chosen: pd.DataFrame, artifacts: Path) -> str:
    """The commands that bring just these checkpoints back from S3 to the folder where MLflow keeps them."""
    return "\n".join(f"aws s3 cp s3://{BUCKET}/artifacts/{path} {artifacts / path}" for path in chosen["checkpoint"])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--results", type=Path, required=True, help="The folder of an experiment in results/experiments, with fold_metrics.csv; the table is written there.")
    parser.add_argument("--models", nargs="+", default=MODELS, help="The models to pick a fold for.")
    parser.add_argument("--manifest", type=Path, default=MLFLOW / "var/mlflow/checkpoint_manifest.csv", help="The manifest written by archive_checkpoints.sh.")
    args = parser.parse_args()
    folds = pd.read_csv(args.results / "fold_metrics.csv", dtype={"fold": str})
    if SCORE not in folds:
        raise SystemExit(f"{SCORE} is not in fold_metrics.csv: export again with src.experiment_table --export")
    chosen = representative(folds, pd.read_csv(args.manifest), args.models)
    missing = chosen[chosen["checkpoint"].isna()]["model"].tolist()
    if missing:
        raise SystemExit(f"No checkpoint in the manifest for {', '.join(missing)}")
    chosen.to_csv(args.results / "representative_checkpoints.csv", index=False)
    print(chosen.to_string(index=False))
    print("\nBring them back from S3:\n" + restore_commands(chosen, MLFLOW / "var/mlflow/artifacts"))


if __name__ == "__main__":
    main()
