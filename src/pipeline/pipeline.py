import json
import pathlib as pl
import time
from importlib import metadata
import subprocess
import tempfile

import mlflow
import mlflow.data
import pandas as pd
import torch
from sklearn.metrics import classification_report, confusion_matrix, roc_curve
from torch.utils.data import DataLoader

from .job import TestJob, TrainJob

from src.results_summary import summarize_metrics
from src.pipeline.metrics import board, class_key
from src.validate_input.params import MODEL_FAMILY
from src.seed import seed_everything
from src.utils import dictionary
from src.dataset import NDBUfesDataset
from src import logger


class Pipeline:
    """Cross-validation over the folds that are not the test fold: one model per validation fold, each tested on the test fold."""

    def __init__(self, params, data_organizer):
        self.params = params
        self.data_organizer = data_organizer
        self.repo = pl.Path(__file__).resolve().parents[2]

    def _dataset(self, arrays, train):
        paths, labels, records = arrays
        return NDBUfesDataset(
            paths,
            labels,
            classes_dict=self.data_organizer.train_classes_dict,
            aug=self.data_organizer.aug if train else None,
            transform=self.data_organizer.train_transform if train else self.data_organizer.test_transform,
            metadata=records,
        )

    def _loaders(self, validation_fold):
        """Training, validation and test loaders. The training order is shuffled with a generator seeded by the seed and the fold."""
        train, validation, test = (self.data_organizer.split(validation_fold))
        batch_size = self.params.hyperparameters.other.batch_size
        generator = torch.Generator()
        generator.manual_seed(self.params.seed + validation_fold)
        return (
            DataLoader(self._dataset(train, train=True), batch_size=batch_size, shuffle=True, generator=generator, num_workers=0, pin_memory=False),
            DataLoader(self._dataset(validation, train=False), batch_size=batch_size, shuffle=False),
            DataLoader(self._dataset(test, train=False), batch_size=batch_size, shuffle=False),
        )

    def train(self):
        """Train one model per validation fold and test each on the test fold. Returns the mean and standard deviation of the metrics over the folds."""
        self.train_job = TrainJob(self.params, self.data_organizer.train_num_classes)
        rows = []
        for fold in self.data_organizer.cv_folds:
            fold_started = time.perf_counter()
            model = self.params.hyperparameters.model.name.value
            with mlflow.start_run(nested=True, run_name=f"{self.params.run_name}_fold_{fold}", tags={"model": model, "model_family": MODEL_FAMILY.get(model, "other"), "validation_fold": str(fold), "test_fold": str(self.data_organizer.test_fold)}):
                logger.info(f"| Fold {fold}: validation | test fold {self.data_organizer.test_fold} |")
                mlflow.log_params({"fold": fold, "fold_seed": self.params.seed + fold})
                seed_everything(self.params.seed + fold)
                self.train_job.checkpoint_description = self._checkpoint_description(fold)
                self._log_json_artifact(self.data_organizer.split_manifest(fold), f"split_manifest_fold_{fold}.json", "split_manifests")
                train_loader, validation_loader, test_loader = self._loaders(fold)

                _, best_validation = self.train_job.run(train_loader, validation_loader, fold=fold)
                best_model = self.train_job.best_model_path
                if self.params.run.checkpoints_in_mlflow:
                    mlflow.log_artifact(best_model, "model")
                mlflow.log_param("model_path", best_model)
                mlflow.log_metrics(board({"checkpoint_mb": pl.Path(best_model).stat().st_size / 2**20}))

                test_job = TestJob(self.params, self.data_organizer.train_num_classes, best_model)
                test_metrics, predictions = test_job.run(test_loader, stage="test")
                class_names = sorted(self.data_organizer.train_classes_dict, key=self.data_organizer.train_classes_dict.get)
                if "histopath_id" in predictions:  # the patch table was given: judge the model per picture and per patient too
                    unit_metrics, unit_tables = summarize_metrics(predictions, class_names, prefix="test")
                    test_metrics |= unit_metrics
                    for unit, table in unit_tables.items():
                        self._log_dataframe_artifact(table, f"{unit}_predictions.csv", "heldout_predictions")
                heldout = {f"heldout_{key}": value for key, value in test_metrics.items()}
                mlflow.log_metrics(board(heldout))
                self._log_prediction_artifacts(predictions, artifact_dir="heldout_predictions")
                self._log_dataframe_artifact(self._roc_curves(predictions, class_names), "roc_curves.csv", "heldout_predictions")
                mlflow.log_metrics(board({"fold_seconds": time.perf_counter() - fold_started}))
                rows.append(best_validation | heldout)
                logger.info(
                    f"Fold {fold} | validation balanced accuracy {best_validation['val_balanced_acc'] * 100:0.3f}% | "
                    f"test balanced accuracy {test_metrics['test_balanced_acc'] * 100:0.3f}%"
                )
            mlflow.log_metrics(board(rows[-1]), step=fold)  # the parent run: one point for each fold

        table = pd.DataFrame(rows)
        summary = {f"avg_{key}": float(table[key].mean()) for key in table} | {f"std_{key}": float(table[key].std(ddof=0)) for key in table}
        mlflow.log_metrics(board(table.mean().to_dict(), "_avg") | board(table.std(ddof=0).to_dict(), "_std"))
        logger.info(
            f"Test balanced accuracy over {len(rows)} folds: {summary['avg_heldout_test_balanced_acc'] * 100:0.3f}% "
            f"+/- {summary['std_heldout_test_balanced_acc'] * 100:0.3f}"
        )
        return summary

    def _checkpoint_description(self, fold):
        organizer = self.data_organizer
        return {
            "model_name": self.params.hyperparameters.model.name.value,
            "training_mode": self.params.training.mode,
            "classes": sorted(organizer.train_classes_dict, key=organizer.train_classes_dict.get),
            "preprocessing": organizer.preprocessing,
            "fold": fold,
            "test_fold": organizer.test_fold,
            "seed": self.params.seed + fold,
            "folds_csv_sha256": organizer.provenance()["folds_csv_sha256"],
        }

    def log_params(self):
        mlflow.log_params(dictionary.flatten(self.params.model_dump()))
        git = {"git_commit": self._run_text(["git", "rev-parse", "HEAD"]), "git_status": self._run_text(["git", "status", "--short"])}
        git["git_dirty"] = bool(git["git_status"]) and not git["git_status"].startswith("unavailable:")
        mlflow.log_params({key: value for key, value in git.items() if len(str(value)) < 500})
        self._log_json_artifact(git, "repo_metadata.json", "provenance")
        versions = {package: metadata.version(package) for package in ("torch", "torchvision", "numpy", "timm", "transformers", "imgaug") if self._installed(package)}
        mlflow.log_params({f"version_{package}": version for package, version in versions.items()})
        provenance = self.data_organizer.provenance() | {"package_versions": versions}
        mlflow.set_tags({"folds_sha256": provenance["folds_csv_sha256"], "git_commit": git["git_commit"], "git_dirty": str(git["git_dirty"])})
        organizer = self.data_organizer
        mlflow.log_input(mlflow.data.from_pandas(organizer.table, source=str(organizer.folds_csv), name="fold_assignments", targets="patch_diagnosis"), context="folds")
        mlflow.log_artifact(str(organizer.folds_csv), "data")
        mlflow.log_params({f"dataset_{key}": value for key, value in provenance.items() if isinstance(value, (str, int, float))})
        self._log_json_artifact(provenance, "dataset_provenance.json", "provenance")

    @staticmethod
    def _installed(package):
        try:
            metadata.version(package)
            return True
        except metadata.PackageNotFoundError:
            return False

    def _run_text(self, command):
        try:
            return subprocess.check_output(command, cwd=self.repo, text=True, stderr=subprocess.STDOUT).strip()
        except Exception as exc:
            return f"unavailable: {exc}"

    @staticmethod
    def _roc_curves(predictions, class_names):
        """The points of the ROC curve of each class against the others."""
        curves = []
        for index, name in enumerate(class_names):
            positive = predictions["y_true"] == index
            if positive.any() and not positive.all():
                fpr, tpr, thresholds = roc_curve(positive, predictions[f"prob_{class_key(name)}"])
                curves.append(pd.DataFrame({"class": name, "fpr": fpr, "tpr": tpr, "threshold": thresholds}))
        return pd.concat(curves, ignore_index=True)

    def _log_prediction_artifacts(self, predictions, artifact_dir):
        self._log_dataframe_artifact(predictions, "predictions.csv", artifact_dir)
        labels = sorted(self.data_organizer.train_classes_dict.values())
        report = classification_report(predictions["y_true"], predictions["y_pred"], labels=labels, output_dict=True, zero_division=0)
        matrix = confusion_matrix(predictions["y_true"], predictions["y_pred"], labels=labels)
        self._log_json_artifact(report, "classification_report.json", artifact_dir)
        self._log_json_artifact({"labels": labels, "matrix": matrix.tolist()}, "confusion_matrix.json", artifact_dir)

    def _log_dataframe_artifact(self, df, filename, artifact_dir):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = pl.Path(temp_dir) / filename
            df.to_csv(path, index=False)
            mlflow.log_artifact(str(path), artifact_path=artifact_dir)

    def _log_json_artifact(self, payload, filename, artifact_dir):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = pl.Path(temp_dir) / filename
            path.write_text(json.dumps(payload, indent=2, default=str))
            mlflow.log_artifact(str(path), artifact_path=artifact_dir)
