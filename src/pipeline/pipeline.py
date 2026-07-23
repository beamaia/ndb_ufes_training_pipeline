from abc import ABC
import json
import os
import pathlib as pl
import subprocess
import tempfile

import mlflow
import numpy as np
import torch
from sklearn.metrics import classification_report, confusion_matrix

from torch.utils.data import DataLoader

from .job import TestJob, TrainJob

from src.utils import dictionary
from src.dataset import NDBUfesDataset
from src import logger

class Pipeline(ABC):
    train_job = None
    test_job = None

    def __init__(self, params, data_organizer):
        self.params = params
        self.data_organizer = data_organizer
        self.train_run_ids = []
        self.best_models = []
        self.reuse_fold_checkpoints = {
            int(fold): pl.Path(path)
            for fold, path in self.params.reuse_fold_checkpoints.items()
        }
        self.training_repo = pl.Path(
            os.environ.get(
                "NDB_UFES_TRAINING_REPO",
                pl.Path(__file__).resolve().parents[2],
            )
        )

    def _inner_train(self, fold_num):
        batch_size = self.params.hyperparameters.other.batch_size

        (train_paths, train_labels, train_metadata), (val_paths, val_labels, val_metadata) = self.data_organizer.data_per_fold(fold=fold_num, train=True)

        train_dataset = NDBUfesDataset(train_paths, train_labels, 
                                       classes_dict=self.data_organizer.train_classes_dict, 
                                       aug=self.data_organizer.aug,
                                       transform=self.data_organizer.train_transform,
                                       metadata=train_metadata)
        val_dataset = NDBUfesDataset(val_paths, val_labels, 
                                     classes_dict=self.data_organizer.train_classes_dict,
                                     aug=None,
                                     transform=self.data_organizer.test_transform,
                                     metadata=val_metadata)
        
        generator = torch.Generator()
        generator.manual_seed(self.params.seed + fold_num)
        train_dataloader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            generator=generator,
            num_workers=0,
            pin_memory=False,
        )
        val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

        metrics, best_loss_metrics = self.train_job.run(train_dataloader, val_dataloader, fold=fold_num)
        return metrics, best_loss_metrics

    def _test_dataloader(self):
        batch_size = self.params.hyperparameters.other.batch_size
        test_paths, test_labels, test_metadata = self.data_organizer.data_per_fold(fold=self.params.dataset.test_fold, train=False)
        test_dataset = NDBUfesDataset(
            test_paths,
            test_labels,
            classes_dict=self.data_organizer.train_classes_dict,
            aug=None,
            transform=self.data_organizer.test_transform,
            metadata=test_metadata,
        )
        return DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    def _validation_dataloader(self, fold_num):
        batch_size = self.params.hyperparameters.other.batch_size
        (_, _, _), (val_paths, val_labels, val_metadata) = self.data_organizer.data_per_fold(
            fold=fold_num,
            train=True,
        )
        val_dataset = NDBUfesDataset(
            val_paths,
            val_labels,
            classes_dict=self.data_organizer.train_classes_dict,
            aug=None,
            transform=self.data_organizer.test_transform,
            metadata=val_metadata,
        )
        return DataLoader(val_dataset, batch_size=batch_size, shuffle=False)


    def train(self):
        train_type = self.params.hyperparameters.other.train_type.value
        self.train_job = TrainJob(self.params, self.data_organizer.train_num_classes)

        if train_type == "holdout":
            with mlflow.start_run(nested=True):
                self._inner_train(self.params.dataset.cv_folds[0])
            
        elif train_type == "cross_validation":
            val_acc_list, val_balanced_acc_list, val_recall_list, val_precision_list = [], [], [], []
            val_macro_f1_list = []
            test_acc_list, test_balanced_acc_list, test_recall_list, test_precision_list = [], [], [], []
            test_macro_f1_list = []

            for fold_num in self.params.dataset.cv_folds:
                with mlflow.start_run(nested=True, run_name=f"fold_{fold_num}"):
                    run = mlflow.active_run()
                    run_id = run.info.run_id
                    self.train_run_ids.append(run_id)
                    
                    logger.info("---------------------------------")
                    logger.info(f"| Starting fold {fold_num} |")
                    mlflow.log_param("fold", fold_num)
                    self._log_split_manifest(fold_num)
                    reused_checkpoint = self.reuse_fold_checkpoints.get(fold_num)
                    if reused_checkpoint:
                        logger.info(f"| Reusing completed checkpoint for fold {fold_num} |")
                        mlflow.log_param("reused_checkpoint", str(reused_checkpoint))
                        mlflow.log_param("training_skipped", True)
                        reused_model = TestJob(
                            self.params,
                            self.data_organizer.train_num_classes,
                            str(reused_checkpoint),
                        )
                        best_val_metrics, _ = reused_model.run(
                            self._validation_dataloader(fold_num),
                            create_artifacts=False,
                            stage="val",
                        )
                        best_model = str(reused_checkpoint)
                    else:
                        _, best_val_metrics = self._inner_train(fold_num)

                        models_path = self.train_job.models_saved
                        best_model = [best for best in models_path if "best.pt" in best and f"fold_{fold_num}" in best]

                        if not len(best_model):
                            raise ValueError("Best model not found.")
                        if len(best_model) > 1:
                            raise TypeError("More than one best model. Type should be string to path not list of paths,")
                        else:
                            best_model = best_model[0]

                    val_acc_list.append(best_val_metrics["val_acc"])
                    val_balanced_acc_list.append(best_val_metrics["val_balanced_acc"])
                    val_recall_list.append(best_val_metrics["val_recall"])
                    val_precision_list.append(best_val_metrics["val_precision"])
                    val_macro_f1_list.append(best_val_metrics["val_macro_f1"])

                    mlflow.log_artifact(best_model, "model")
                    mlflow.log_param("model_path", best_model)
                    self.best_models.append({"fold": fold_num, "path": best_model, "run_id": run_id})

                    test_metrics, test_predictions = self._evaluate_model_on_test(best_model)
                    prefixed_test_metrics = {f"heldout_{key}": item for key, item in test_metrics.items()}
                    mlflow.log_metrics(prefixed_test_metrics)
                    self._log_prediction_artifacts(test_predictions, artifact_dir="heldout_predictions")

                    test_acc_list.append(test_metrics["test_acc"])
                    test_balanced_acc_list.append(test_metrics["test_balanced_acc"])
                    test_recall_list.append(test_metrics["test_recall"])
                    test_precision_list.append(test_metrics["test_precision"])
                    test_macro_f1_list.append(test_metrics["test_macro_f1"])

                    logger.info(f"| Finished training fold {fold_num}|")
                    logger.info(f"Best model metrics | Accuracy: {best_val_metrics['val_acc'] * 100: 0.3f}% | Balanced accuracy: {best_val_metrics['val_balanced_acc'] * 100: 0.3f}% | Macro F1: {best_val_metrics['val_macro_f1'] * 100: 0.3f}% | Recall: {best_val_metrics['val_recall'] * 100: 0.3f}% | Precision {best_val_metrics['val_precision'] * 100: 0.3f}%")
                    logger.info("---------------------------------\n")
                    
                metrics = {f"fold_{key}": item for key, item in best_val_metrics.items()}

                mlflow.log_metrics(metrics, step=fold_num)
                
            avg_std_metrics = {
                "avg_val_acc": np.mean(val_acc_list), 
                "avg_val_balanced_acc": np.mean(val_balanced_acc_list), 
                "avg_val_recall": np.mean(val_recall_list), 
                "avg_val_precision": np.mean(val_precision_list), 
                "avg_val_macro_f1": np.mean(val_macro_f1_list),
                "std_val_acc": np.std(val_acc_list), 
                "std_val_balanced_acc": np.std(val_balanced_acc_list), 
                "std_val_recall": np.std(val_recall_list), 
                "std_val_precision": np.std(val_precision_list), 
                "std_val_macro_f1": np.std(val_macro_f1_list),
                "avg_heldout_test_acc": np.mean(test_acc_list),
                "avg_heldout_test_balanced_acc": np.mean(test_balanced_acc_list),
                "avg_heldout_test_recall": np.mean(test_recall_list),
                "avg_heldout_test_precision": np.mean(test_precision_list),
                "avg_heldout_test_macro_f1": np.mean(test_macro_f1_list),
                "std_heldout_test_acc": np.std(test_acc_list),
                "std_heldout_test_balanced_acc": np.std(test_balanced_acc_list),
                "std_heldout_test_recall": np.std(test_recall_list),
                "std_heldout_test_precision": np.std(test_precision_list),
                "std_heldout_test_macro_f1": np.std(test_macro_f1_list),
            }
            mlflow.log_metrics(avg_std_metrics)
            logger.info(f"Average validation accuracy: {avg_std_metrics['avg_val_acc'] * 100: 0.3f}% +/- {avg_std_metrics['std_val_acc'] * 100: 0.3f}")
            logger.info(f"Average held-out test accuracy: {avg_std_metrics['avg_heldout_test_acc'] * 100: 0.3f}% +/- {avg_std_metrics['std_heldout_test_acc'] * 100: 0.3f}")

        else:
            raise ValueError(f"String value {train_type} invalid for chosing training type of job.")   
        
            
    def test(self):
        if not self.best_models:
            logger.info("No trained CV checkpoints found in this process; held-out testing is performed during training.")


    def origin_test(self):
        pass


    def log_params(self):
        params_dict = self.params.model_dump()
        params_dict_flatten = dictionary.flatten(params_dict)

        mlflow.log_params(params_dict_flatten)
        self._log_repo_metadata()
        self._log_dataset_provenance()

    def _evaluate_model_on_test(self, model_path):
        test_job = TestJob(self.params, self.data_organizer.train_num_classes, model_path)
        metrics, predictions = test_job.run(
            self._test_dataloader(),
            create_artifacts=False,
            stage="test",
        )
        return metrics, predictions

    def _log_split_manifest(self, fold_num):
        manifest = self.data_organizer.split_manifest(fold_num)
        self._log_json_artifact(manifest, f"split_manifest_fold_{fold_num}.json", "split_manifests")

    def _log_dataset_provenance(self):
        provenance = self.data_organizer.provenance()
        for key, value in provenance.items():
            if isinstance(value, (str, int, float)):
                mlflow.log_param(f"dataset_{key}", value)
        self._log_json_artifact(provenance, "dataset_provenance.json", "provenance")

    def _log_repo_metadata(self):
        metadata = {
            "git_commit": self._run_text(["git", "rev-parse", "HEAD"]),
            "git_status": self._run_text(["git", "status", "--short"]),
            "dvc_status": self._run_text(["dvc", "status"]),
        }
        git_status = metadata["git_status"]
        metadata["git_dirty"] = bool(
            git_status and not git_status.startswith("unavailable:")
        )
        for key, value in metadata.items():
            if isinstance(value, str) and len(value) < 500:
                mlflow.log_param(key, value)
        self._log_json_artifact(metadata, "repo_metadata.json", "provenance")

    def _run_text(self, cmd):
        try:
            return subprocess.check_output(
                cmd,
                cwd=self.training_repo,
                text=True,
                stderr=subprocess.STDOUT,
            ).strip()
        except Exception as exc:
            return f"unavailable: {exc}"

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
        
