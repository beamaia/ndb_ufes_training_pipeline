from abc import ABC
from enum import Enum

import mlflow
import numpy as np

from torch.utils.data import DataLoader

from .job import TestJob, TrainJob

from src.utils import dictionary
from src.dataset import NDBUfesDataset
from src import logger

class Pipeline(ABC):
    train_job = None
    test_job = None

    train_run_ids =  []

    def __init__(self, params, data_organizer):
        self.params = params
        self.data_organizer = data_organizer

    def _inner_train(self, fold_num):
        batch_size = self.params.hyperparameters.other.batch_size

        (train_paths, train_labels), (val_paths, val_labels) = self.data_organizer.data_per_fold(fold=fold_num, train=True)

        train_dataset = NDBUfesDataset(train_paths, train_labels, 
                                       classes_dict=self.data_organizer.train_classes_dict, 
                                       transform=self.data_organizer.train_transform)
        val_dataset = NDBUfesDataset(val_paths, val_labels, 
                                     classes_dict=self.data_organizer.train_classes_dict,
                                     transform=self.data_organizer.test_transform)
        
        train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)

        metrics, best_loss_metrics = self.train_job.run(train_dataloader, val_dataloader, fold=fold_num)
        return metrics, best_loss_metrics


    def train(self):
        train_type = self.params.hyperparameters.other.train_type.value
        self.train_job = TrainJob(self.params, self.data_organizer.train_num_classes)

        if train_type == "holdout":
            with mlflow.start_run(nested=True):
                fold_num = 0
                self._inner_train(fold_num)
            
        elif train_type == "cross_validation":
            num_folds = self.params.hyperparameters.other.folds
            val_acc_list, val_balanced_acc_list, val_recall_list, val_precision_list = [], [], [], []

            for fold_num in range(1, num_folds + 1):
                with mlflow.start_run(nested=True, run_name=f"fold_{fold_num}"):
                    run = mlflow.active_run()
                    run_id = run.info.run_id
                    self.train_run_ids.append(run_id)
                    
                    logger.info("---------------------------------")
                    logger.info(f"| Starting fold {fold_num} |")
                    mlflow.log_param("fold", fold_num)
                    _, best_val_metrics = self._inner_train(fold_num)

                    val_acc_list.append(best_val_metrics["val_acc"])
                    val_balanced_acc_list.append(best_val_metrics["val_balanced_acc"])
                    val_recall_list.append(best_val_metrics["val_recall"])
                    val_precision_list.append(best_val_metrics["val_precision"])

                    models_path = self.train_job.models_saved
                    best_model = [best for best in models_path if "best.pt" in best and f"fold{fold_num}" in best]
                    
                    if not len(best_model):
                        raise ValueError("Best model not found.")
                    if len(best_model) > 1:
                        raise TypeError("More than one best model. Type should be string to path not list of paths,")
                    else:
                        best_model = best_model[0]
                    
                    mlflow.log_artifact(best_model, "model")
                    mlflow.log_param("model_path", best_model)

                    logger.info(f"| Finished training fold {fold_num}|")
                    logger.info(f"Best model metrics | Accuracy: {best_val_metrics["val_acc"] * 100: 0.3f}% | Balanced accuracy: {best_val_metrics["val_balanced_acc"] * 100: 0.3f}% | Recall: {best_val_metrics["val_recall"] * 100: 0.3f}% | Precision {best_val_metrics["val_precision"] * 100: 0.3f}%")
                    logger.info("---------------------------------\n")
                    
                metrics = {f"fold_{key}": item for key, item in best_val_metrics.items()}

                mlflow.log_metrics(metrics, step=fold_num)
                
            avg_std_metrics = {
                "avg_val_acc": np.mean(val_acc_list), 
                "avg_val_balanced_acc": np.mean(val_balanced_acc_list), 
                "avg_val_recall": np.mean(val_recall_list), 
                "avg_val_precision": np.mean(val_precision_list), 
                "std_val_acc": np.std(val_acc_list), 
                "std_val_balanced_acc": np.std(val_balanced_acc_list), 
                "std_val_recall": np.std(val_recall_list), 
                "std_val_precision": np.std(val_precision_list), 
            }
            mlflow.log_metrics(avg_std_metrics)
            logger.info(f"Average metrics during training | Accuracy: {avg_std_metrics["avg_val_acc"] * 100: 0.3f}% \u00B12 {avg_std_metrics["std_val_acc"] * 100: 0.3f}")
            logger.info(f"Average metrics during training | Balanced Accuracy: {avg_std_metrics["avg_val_balanced_acc"] * 100: 0.3f}% \u00B12 {avg_std_metrics["std_val_balanced_acc"] * 100: 0.3f}")
            logger.info(f"Average metrics during training | Recall: {avg_std_metrics["avg_val_recall"] * 100: 0.3f}% \u00B12 {avg_std_metrics["std_val_recall"] * 100: 0.3f}")
            logger.info(f"Average metrics during training | Precision: {avg_std_metrics["avg_val_precision"] * 100: 0.3f}% \u00B12 {avg_std_metrics["std_val_precision"] * 100: 0.3f}")

        else:
            raise ValueError(f"String value {train_type} invalid for chosing training type of job.")   
        
            
    def test(self):
        if self.params.stages.train:
            (test_paths, test_labels) = self.data_organizer.data_per_fold(fold=0, train=False)

            for run_id in self.train_run_ids:
                run = mlflow.search_runs()
            

    def origin_test(self):
        pass


    def log_params(self):
        params_dict = self.params.model_dump()
        params_dict_flatten = dictionary.flatten(params_dict)

        mlflow.log_params(params_dict_flatten)
        
