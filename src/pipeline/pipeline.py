from abc import ABC
from enum import Enum

import mlflow

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

        self.train_job.run(train_dataloader, val_dataloader, fold=fold_num)


    def train(self):
        train_type = self.params.hyperparameters.other.train_type.value
        self.train_job = TrainJob(self.params, self.data_organizer.train_num_classes)

        if train_type == "holdout":
            fold_num = 0
            self._inner_train(fold_num)
            
        elif train_type == "cross_validation":
            num_folds = self.params.hyperparameters.other.folds

            for fold_num in range(1, num_folds + 1):
                logger.info(f"| Starting fold {fold_num} |")
                self._inner_train(fold_num)
                logger.info(f"| Finished training fold {fold_num}|")

        else:
            raise ValueError(f"String value {train_type} invalid for chosing training type of job.")   
        
            
    def test(self):
        pass

    def origin_test(self):
        pass


    def log_params(self):
        params_dict = self.params.model_dump()
        params_dict_flatten = dictionary.flatten(params_dict)

        mlflow.log_params(params_dict_flatten)
        
