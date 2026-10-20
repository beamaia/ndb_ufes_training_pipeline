import argparse
import os
import platform
import subprocess
import time
from pathlib import Path

import mlflow
import torch
import yaml
from dotenv import dotenv_values

REPO = Path(__file__).resolve().parent


def load_env(*files: Path) -> None:
    """Read only the MLflow credentials and the Hugging Face token from the .env files; the other variables there (cloud keys, for example) stay out of this process."""
    for file in files:
        if file.exists():
            for key, value in dotenv_values(file).items():
                if value and (key.startswith("MLFLOW_") or key == "HF_TOKEN"):
                    os.environ.setdefault(key, value)


load_env(REPO / ".env", Path(os.getenv("MLFLOW_ENV_FILE", REPO.parent / "ndb_ufes_mlflow" / ".env")))

from src.config import load_params
from src.dataset import LeakageSafeNDBUfesOrganizer
from src.pipeline import Pipeline
from src.validate_input.params import MODEL_FAMILY
from src import logger, log_file


def arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--params", type=Path, nargs="+", default=[REPO / "params.yaml"], help="YAML files with the settings. Each file is one run, and the runs go one after the other. A file can extend another one (extends: params.yaml).")
    parser.add_argument("--check-data", action="store_true", help="Check the folds file and the images, then stop.")
    parser.add_argument("--resume", action="store_true", help="Continue the folds that were interrupted.")
    return parser


def hardware(device: str) -> dict[str, str]:
    """Where the run happens, as MLflow tags."""
    if device == "cuda":
        name = torch.cuda.get_device_name(0)
    elif platform.system() == "Darwin":
        name = subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True, text=True).stdout.strip() or platform.processor()
        name += " (MPS)" if device == "mps" else ""
    else:
        name = platform.processor() or platform.machine()
    return {"device": device, "device_name": name, "host": "runpod" if os.getenv("RUNPOD_POD_ID") else "local", "platform": platform.platform()}


def main() -> None:
    args = arg_parser().parse_args()
    runs = [load_params(file, REPO, resume=args.resume) for file in args.params]  # every file is checked before the first run starts
    checked = set()  # the datasets already checked in this run
    for params, settings_files in runs:
        logger.info(f"Settings file: {settings_files[-1]}")
        run(params, settings_files, args.check_data, checked)


def run(params, settings_files, check_data: bool, checked: set) -> None:
    """The runs of one settings file: every model in it, one after the other."""
    models = params.experiment.models or [params.hyperparameters.model.name]

    first = params.model_copy(deep=True)
    first.hyperparameters.model.name = models[0]
    dataset = params.dataset
    dataset_key = (dataset.folds_csv, dataset.patch_dir, dataset.patch_table, tuple(dataset.folds), dataset.test_fold)
    organizer = LeakageSafeNDBUfesOrganizer(first, check=dataset_key not in checked)
    if dataset_key in checked:
        logger.info("Data already checked in this run")
    else:
        missing = organizer.missing_images()
        if missing:
            raise FileNotFoundError(f"{len(missing)} patch images are missing in {dataset.patch_dir}, for example {missing[:5]}")
        checked.add(dataset_key)
        logger.info(f"Data check OK: {organizer.provenance()}")
    if check_data:
        return

    mlflow.set_tracking_uri(params.run.tracking_uri or os.getenv("MLFLOW_TRACKING_URI", "http://localhost:8000/"))
    experiment = mlflow.set_experiment(f"ndb_ufes_{params.run.purpose}")
    try:
        import psutil  # noqa: F401  (system metrics need it)
        system_metrics = True
    except ImportError:
        system_metrics = False
    for model in models:
        run_params = params.model_copy(deep=True)
        run_params.hyperparameters.model.name = model
        optimizer = run_params.hyperparameters.optimizer.name.value
        run_params.run_name = "_".join(part for part in (model.value, optimizer, params.run.suffix) if part)
        logger.info(run_params)
        tags = {
            "model": model.value, "model_family": MODEL_FAMILY.get(model.value, "other"), "optimizer": optimizer, "task": "multiclass", "purpose": params.run.purpose,
            "test_fold": str(run_params.dataset.test_fold), "seed": str(run_params.seed), "epochs": str(run_params.hyperparameters.other.epochs),
            "batch_size": str(run_params.hyperparameters.other.batch_size), **hardware(run_params.device),
        }
        with mlflow.start_run(
            run_name=run_params.run_name,
            experiment_id=experiment.experiment_id,
            tags=tags,
            description="Multiclass patch classification, five cross-validation folds, every fold model tested on the test fold.",
            log_system_metrics=system_metrics,
        ):
            started = time.perf_counter()
            for file in settings_files:  # the file, the ones it extends, and what they add up to
                mlflow.log_artifact(str(file), "config")
            mlflow.log_text(yaml.safe_dump(run_params.model_dump(mode="json"), sort_keys=False), "config/merged_params.yaml")
            pipeline = Pipeline(run_params, LeakageSafeNDBUfesOrganizer(run_params, check=False))  # the data was checked above
            pipeline.log_params()
            pipeline.train()
            seconds = time.perf_counter() - started
            mlflow.log_metric("run_seconds", seconds)
            if params.run.hourly_cost is not None:
                mlflow.log_metric("cost_usd", seconds / 3600 * params.run.hourly_cost)
            mlflow.log_artifact(log_file)


if __name__ == "__main__":
    main()
