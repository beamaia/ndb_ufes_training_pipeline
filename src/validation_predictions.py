import argparse
import os
import tempfile
from pathlib import Path

import mlflow
import pandas as pd

from src import logger
from src.config import load_env, load_params, resolve_device
from src.dataset import LeakageSafeNDBUfesOrganizer
from src.pipeline import Pipeline
from src.pipeline.job import TestJob
from src.validate_input.params import ModelEnum

REPO = Path(__file__).resolve().parents[1]


def best_checkpoint(run_id: str, directory: Path) -> Path:
    """Download the best checkpoint of a run from MLflow."""
    names = [item.path for item in mlflow.artifacts.list_artifacts(run_id=run_id, artifact_path="model") if item.path.endswith("_best.pt")]
    if len(names) != 1:
        raise FileNotFoundError(f"Run {run_id} has {len(names)} best checkpoints in MLflow, not one")
    return Path(mlflow.artifacts.download_artifacts(run_id=run_id, artifact_path=names[0], dst_path=str(directory)))


def validation_predictions(params, fold: int, checkpoint: Path) -> pd.DataFrame:
    """The predictions of the model in ``checkpoint`` on validation fold ``fold``, from the loaders and the test job that the training itself uses."""
    organizer = LeakageSafeNDBUfesOrganizer(params, check=False)
    validation_loader = Pipeline(params, organizer).loaders(fold)[1]
    _, predictions = TestJob(params, organizer.train_num_classes, str(checkpoint)).run(validation_loader, stage="val")
    return predictions.assign(validation_fold=fold)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--experiment", required=True, help="The MLflow experiment, for example ndb_ufes_exp2.")
    parser.add_argument("--folds-csv", type=Path, help="The folds file the experiment used (default: the new folds of the organizer).")
    parser.add_argument("--models", nargs="+", help="Only these models (default: all).")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output", type=Path, help="CSV to write (default: notebooks/cache/<experiment>_validation.csv).")
    args = parser.parse_args()

    load_env(REPO / ".env", Path(os.getenv("MLFLOW_ENV_FILE", REPO.parent / "ndb_ufes_mlflow" / ".env")))
    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://localhost:8000/"))
    params, _ = load_params(REPO / "params.yaml", REPO)
    if args.folds_csv:
        params.dataset.folds_csv = str(args.folds_csv.resolve())
    params.run.num_workers = 0
    device = resolve_device(args.device)
    output = args.output or REPO / "notebooks" / "cache" / f"{args.experiment}_validation.csv"

    runs = mlflow.search_runs(experiment_names=[args.experiment], filter_string="attributes.status = 'FINISHED'")
    folds = runs[runs["tags.mlflow.parentRunId"].notna()].sort_values(["tags.model", "tags.validation_fold"])
    if args.models:
        folds = folds[folds["tags.model"].isin(args.models)]
    on_cpu, frames = set(), []  # the models that could not run on the device (some operations of a model may not exist on a Mac GPU)
    for _, run in folds.iterrows():
        model, fold = run["tags.model"], int(run["tags.validation_fold"])
        run_params = params.model_copy(deep=True)
        run_params.hyperparameters.model.name = ModelEnum(model)
        with tempfile.TemporaryDirectory() as scratch:
            checkpoint = best_checkpoint(run["run_id"], Path(scratch))
            for where in ("cpu",) if model in on_cpu or device == "cpu" else (device, "cpu"):
                run_params.device = where
                try:
                    predictions = validation_predictions(run_params, fold, checkpoint)
                    break
                except RuntimeError as error:
                    if where == "cpu":
                        raise
                    logger.warning(f"{model} could not run on {where} ({str(error)[:80]}...); it goes on the cpu")
                    on_cpu.add(model)
        frames.append(predictions)
        logger.info(f"{model} fold {fold}: {len(predictions)} validation patches")
    output.parent.mkdir(parents=True, exist_ok=True)
    pd.concat(frames, ignore_index=True).to_csv(output, index=False)
    logger.info(f"Wrote {output}")


if __name__ == "__main__":
    main()
