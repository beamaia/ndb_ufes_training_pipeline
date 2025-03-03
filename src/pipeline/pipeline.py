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

        train_acc, val_acc = self.train_job.run(train_dataloader, val_dataloader, fold=fold_num)
        return train_acc, val_acc


    def train(self):
        train_type = self.params.hyperparameters.other.train_type.value
        self.train_job = TrainJob(self.params, self.data_organizer.train_num_classes)

        if train_type == "holdout":
            with mlflow.start_run(nested=True):
                fold_num = 0
                self._inner_train(fold_num)
            
        elif train_type == "cross_validation":
            num_folds = self.params.hyperparameters.other.folds
            train_acc_list, val_acc_list = [], []
            
            for fold_num in range(1, num_folds + 1):
                with mlflow.start_run(nested=True, run_name=f"fold_{fold_num}"):
                    run = mlflow.active_run()
                    run_id = run.info.run_id
                    self.train_run_ids.append(run_id)

                    logger.info(f"| Starting fold {fold_num} |")
                    mlflow.log_param("fold", fold_num)
                    train_acc, val_acc = self._inner_train(fold_num)

                    train_acc_list.append(train_acc)
                    val_acc_list.append(val_acc)

                    models_path = self.train_job.models_saved
                    best_model = [best for best in models_path if "best.pt" in best and f"fold{fold_num}" in best]
                    print(models_path)
                    
                    if not len(best_model):
                        raise ValueError("Best model not found.")
                    if len(best_model) > 1:
                        raise TypeError("More than one best model. Type should be string to path not list of paths,")
                    else:
                        best_model = best_model[0]
                    
                    mlflow.log_artifact(best_model, "model")
                    mlflow.log_param("model_path", best_model)

                    logger.info(f"| Finished training fold {fold_num}|")

                mlflow.log_metrics({
                    "val_accuracy": val_acc,
                }, step=fold_num)
                

            mlflow.log_metrics({
                "avg_train_accuracy": np.mean(train_acc_list), 
                "avg_val_accuracy": np.mean(val_acc_list)
            })
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
        
