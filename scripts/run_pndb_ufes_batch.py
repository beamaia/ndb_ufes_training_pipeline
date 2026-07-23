#!/usr/bin/env python3
"""Run one P-NDB-UFES batch through the training pipeline."""

from __future__ import annotations

import argparse
import os
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml


TRAINING_REPO = Path(__file__).resolve().parents[1]
THESIS_REPO = Path(os.environ.get(
    "NDB_UFES_THESIS_REPO",
    str(TRAINING_REPO.parent / "ndb_ufes_data_organizer"),
))
ORGANIZER_DATA = Path(os.environ.get(
    "NDB_UFES_ORGANIZER_DATA",
    str(THESIS_REPO / "data"),
))
OUTPUT_ROOT = Path(os.environ.get(
    "NDB_UFES_OUTPUT_ROOT",
    str(THESIS_REPO / "results/training_runs"),
))
BATCH_CSVS = {
    "batch1": THESIS_REPO / "results/phase3/current_thesis_batches/batch1_recovered_reference_patch_level.csv",
    "batch2": THESIS_REPO / "results/phase3/current_thesis_batches/batch2_patient_first_patch_level.csv",
    "batch3": THESIS_REPO / "results/phase3/current_thesis_batches/batch3_virchow_pruned_patch_level.csv",
}
CANONICAL_BATCHES = ("batch1", "batch2")
CANONICAL_MODELS = ("mobilenetv2", "densenet121", "resnet50")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", choices=sorted(BATCH_CSVS), required=True)
    parser.add_argument("--models", nargs="+", default=list(CANONICAL_MODELS))
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--batch-size", type=int, default=30)
    parser.add_argument("--optimizer", choices=["sgd", "adam"], default=None)
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--momentum", type=float, default=None)
    parser.add_argument("--scheduler-patience", type=int, default=None)
    parser.add_argument("--scheduler-factor", type=float, default=None)
    parser.add_argument("--scheduler-min-lr", type=float, default=None)
    parser.add_argument("--early-stopping-patience", type=int, default=None)
    parser.add_argument(
        "--device",
        default="auto",
        help="Torch device. 'auto' uses MPS when available and otherwise CPU.",
    )
    parser.add_argument("--node-type", default="float32")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--torch-threads",
        type=int,
        default=None,
        help="Optional torch CPU thread count; useful on Apple Silicon CPU fallback.",
    )
    parser.add_argument("--run-suffix", default="smoke")
    parser.add_argument(
        "--resume",
        action="store_true",
        help=(
            "Resume interrupted folds for the same batch/model/run suffix. "
            "Requires a training-state checkpoint created by this pipeline."
        ),
    )
    parser.add_argument(
        "--cv-folds",
        nargs="+",
        type=int,
        default=None,
        help="Cross-validation folds to run. The held-out test fold remains unchanged.",
    )
    parser.add_argument(
        "--reuse-folds",
        nargs="+",
        type=int,
        default=None,
        help="Folds to evaluate from completed checkpoints instead of retraining.",
    )
    parser.add_argument(
        "--reuse-from-run-name",
        default=None,
        help="Existing run name used to locate completed fold checkpoints.",
    )
    parser.add_argument("--tracking-uri", default=None)
    parser.add_argument(
        "--max-rows-per-fold",
        type=int,
        default=None,
        help="Optional smoke-test limit; never use this for final experiments.",
    )
    parser.add_argument(
        "--no-class-weights",
        action="store_true",
        help="Disable class weights for tiny smoke tests only.",
    )
    parser.add_argument("--check-data-only", action="store_true")
    parser.add_argument(
        "--allow-exploratory-batch3",
        action="store_true",
        help="Explicitly allow the archived Virchow-pruned Batch 3 workflow.",
    )
    args = parser.parse_args()
    if args.batch == "batch3" and not args.allow_exploratory_batch3:
        parser.error(
            "Batch 3 is exploratory and is not part of the v1.0.0 thesis "
            "comparison; pass --allow-exploratory-batch3 to run it explicitly."
        )
    return args


def main() -> None:
    args = parse_args()
    sys.path.insert(0, str(TRAINING_REPO))
    os.environ.setdefault("TORCH_HOME", str(TRAINING_REPO / ".cache/torch"))
    try:
        import certifi
        os.environ.setdefault("SSL_CERT_FILE", certifi.where())
    except ImportError:
        pass

    import torch
    from src.dataset import LeakageSafeNDBUfesOrganizer
    from src.pipeline import Pipeline
    from src.validate_input.params import Parameters

    params_dict = yaml.safe_load((TRAINING_REPO / "params.yaml").read_text())
    params_dict["run_name"] = f"{args.batch}_{args.run_suffix}"
    params_dict["dataset"]["root"] = str(ORGANIZER_DATA)
    params_dict["dataset"]["fold_assignments_path"] = str(BATCH_CSVS[args.batch])
    params_dict["dataset"]["group_column"] = "patient_case_group"
    params_dict["dataset"]["allow_origin_overlap"] = args.batch == "batch1"
    params_dict["dataset"]["allow_group_overlap"] = args.batch == "batch1"
    params_dict["experiment"]["models"] = args.models
    params_dict["hyperparameters"]["model"]["name"] = args.models[0]
    params_dict["hyperparameters"]["other"]["epochs"] = args.epochs
    params_dict["hyperparameters"]["other"]["batch_size"] = args.batch_size
    if args.optimizer is not None:
        params_dict["hyperparameters"]["optimizer"]["name"] = args.optimizer
    if args.learning_rate is not None:
        params_dict["hyperparameters"]["optimizer"]["learning_rate"] = args.learning_rate
    if args.momentum is not None:
        params_dict["hyperparameters"]["optimizer"]["momentum"] = args.momentum
    scheduler_params = params_dict["hyperparameters"]["scheduler"]["other"]
    if args.scheduler_patience is not None:
        scheduler_params["patience"] = args.scheduler_patience
    if args.scheduler_factor is not None:
        scheduler_params["factor"] = args.scheduler_factor
    if args.scheduler_min_lr is not None:
        scheduler_params["min_lr"] = args.scheduler_min_lr
    if args.early_stopping_patience is not None:
        params_dict["early_stopping"]["patience"] = args.early_stopping_patience
    if args.seed < 0:
        raise ValueError("--seed must be non-negative.")
    params_dict["seed"] = args.seed
    params_dict["resume"] = args.resume
    if args.cv_folds is not None:
        params_dict["dataset"]["cv_folds"] = args.cv_folds
    if args.reuse_folds:
        if not args.reuse_from_run_name:
            raise ValueError("--reuse-folds requires --reuse-from-run-name.")
        if len(args.models) != 1:
            raise ValueError("--reuse-folds supports one model per invocation.")
        optimizer_name = params_dict["hyperparameters"]["optimizer"]["name"]
        checkpoint_root = (
            OUTPUT_ROOT / args.batch / "models"
            / f"project_multiclass_run_{args.reuse_from_run_name}"
        )
        reused = {}
        for fold in args.reuse_folds:
            checkpoint_path = checkpoint_root / f"fold_{fold}" / (
                f"model_{args.models[0]}_optim_{optimizer_name}_best.pt"
            )
            if not checkpoint_path.exists():
                raise FileNotFoundError(f"Completed fold checkpoint not found: {checkpoint_path}")
            reused[str(fold)] = str(checkpoint_path)
        params_dict["reuse_fold_checkpoints"] = reused
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.no_class_weights:
        params_dict["hyperparameters"]["other"]["loss_weights"] = False
    if args.torch_threads is not None:
        if args.torch_threads < 1:
            raise ValueError("--torch-threads must be positive.")
        torch.set_num_threads(args.torch_threads)
        print(f"Torch CPU threads: {torch.get_num_threads()}")

    requested_device = args.device.lower()
    if requested_device == "auto":
        resolved_device = "mps" if torch.backends.mps.is_available() else "cpu"
        if resolved_device == "cpu":
            print("MPS is unavailable; using CPU fallback.")
    elif requested_device == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError(
            "MPS was requested but is unavailable in this PyTorch/macOS environment. "
            "Use --device cpu or --device auto."
        )
    else:
        resolved_device = requested_device
    params_dict["device"] = resolved_device
    params_dict["node_type"] = args.node_type

    # OUTPUT_ROOT already names the training-run root. Keep one directory per batch.
    run_dir = OUTPUT_ROOT / args.batch
    if not args.check_data_only or args.max_rows_per_fold is not None:
        run_dir.mkdir(parents=True, exist_ok=True)
        os.chdir(run_dir)

    if args.max_rows_per_fold is not None:
        if args.max_rows_per_fold < 1:
            raise ValueError("--max-rows-per-fold must be positive.")
        source_table = pd.read_csv(BATCH_CSVS[args.batch])
        smoke_table = source_table.groupby("fold", group_keys=False).head(args.max_rows_per_fold)
        smoke_csv = run_dir / f"smoke_fold_assignments_{args.run_suffix}.csv"
        smoke_table.to_csv(smoke_csv, index=False)
        params_dict["dataset"]["fold_assignments_path"] = str(smoke_csv)
        print(f"Smoke-test fold CSV: {smoke_csv}")

    params_model = Parameters.model_validate(params_dict)
    first_model = params_model.model_copy(deep=True)
    first_model.hyperparameters.model.name = params_model.experiment.models[0]
    organizer = LeakageSafeNDBUfesOrganizer(first_model)
    (train_paths, _, train_meta), (val_paths, _, val_meta) = organizer.data_per_fold(
        params_model.dataset.cv_folds[0], train=True
    )
    test_paths, _, test_meta = organizer.data_per_fold(params_model.dataset.test_fold, train=False)
    missing = [
        str(path)
        for path in list(train_paths) + list(val_paths) + list(test_paths)
        if not Path(path).exists()
    ]
    if missing:
        raise FileNotFoundError(f"Missing configured images: {missing[:5]}")
    print(f"Data check OK for {args.batch}")
    print(f"CSV: {params_model.dataset.fold_assignments_path}")
    print(f"First split sizes: train={len(train_paths)}, val={len(val_paths)}, test={len(test_paths)}")
    print(f"Train folds: {sorted({row['fold'] for row in train_meta})}")
    print(f"Validation folds: {sorted({row['fold'] for row in val_meta})}")
    print(f"Test folds: {sorted({row['fold'] for row in test_meta})}")
    if args.check_data_only:
        return

    import mlflow
    from dotenv import load_dotenv

    mlflow_env = Path(os.environ.get(
        "NDB_UFES_MLFLOW_ENV",
        str(TRAINING_REPO.parent / "ndb_ufes_mlflow/.env"),
    ))
    load_dotenv(mlflow_env, override=False)
    tracking_uri = args.tracking_uri or os.getenv("MLFLOW_TRACKING_URI")
    if not tracking_uri:
        tracking_uri = "http://localhost:8000/"
    mlflow.set_tracking_uri(tracking_uri)
    print(f"MLflow tracking URI: {tracking_uri}")
    experiment = mlflow.set_experiment("pndb_ufes_leakage_batches")

    for model_name in params_model.experiment.models:
        run_params = params_model.model_copy(deep=True)
        run_params.hyperparameters.model.name = model_name
        run_params.run_name = f"{args.batch}_{model_name.value}_{args.run_suffix}"
        data_organizer = LeakageSafeNDBUfesOrganizer(run_params)
        with mlflow.start_run(
            run_name=run_params.run_name,
            experiment_id=experiment.experiment_id,
            tags={"batch": args.batch, "model": model_name.value, "task": "multiclass"},
            description=(
                "P-NDB-UFES original-comparable split experiment."
                if args.batch == "batch1"
                else "P-NDB-UFES lower-contamination-risk grouped split experiment."
            ),
        ):
            pipeline = Pipeline(run_params, data_organizer)
            pipeline.log_params()
            pipeline.train()
            pipeline.test()


if __name__ == "__main__":
    main()
