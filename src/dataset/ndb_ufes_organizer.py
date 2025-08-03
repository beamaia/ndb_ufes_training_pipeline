from abc import ABC
import pathlib as pl

import pandas as pd

from torchvision import transforms
from torchvision.transforms import v2
from torch import bfloat16
import torch
from imgaug import augmenters as iaa


from src import logger

class NDBUfesOrganizer(ABC):
    oscc_bin_root = "bin_oscc_not_oscc_folds"
    dys_bin_root =  "bin_dysplasia_no_dysplasia_folds"
    oscc_dys_root = "bin_oscc_dysplasia_folds"
    multiclass_root = "multiclass_folds"
    
    class DataTable(ABC):
        def __init__(self, root_path, folds_division_path):
            self.train = pd.read_csv(root_path / f"{folds_division_path}_train.csv")
            self.test = pd.read_csv(root_path / f"{folds_division_path}_test.csv")
            self.origin_test = pd.read_csv(root_path / f"origin_test.csv")
            
    def __init__(self, patch: str, origin: str, root: str, task: str, node_type:str):
        self.root_path = pl.Path(root)
        self.patch_path = pl.Path(self.root_path / patch)
        self.origin_path = pl.Path(self.root_path / origin)
        self.task = task
        
        self.folds_division_path = self._define_task_file()
        self.train_classes_dict = self._define_classes(self.task)
        self.train_num_classes = len(self.train_classes_dict)

        self.data = self.DataTable(self.root_path, self.folds_division_path)
        self.origin_classes_dict = self._define_classes("multiclass")

        self.node_type = getattr(torch, node_type)
        self.aug = self._configure_aug((224, 224))

        self.train_transform = transforms.Compose([transforms.ToTensor(),
                                                   transforms.ConvertImageDtype(self.node_type),
        ])

        self.test_transform = transforms.Compose([transforms.ToTensor(),
                                                  transforms.ConvertImageDtype(self.node_type)])

    def _configure_aug(self, size):
        return iaa.Sequential([
            iaa.Sometimes(0.25, iaa.Affine(scale={"x": (1.0, 2.0), "y": (1.0, 2.0)})),
            iaa.Scale(size),
            iaa.Fliplr(0.5),
            iaa.Flipud(0.2),
            iaa.Sometimes(0.25, iaa.Affine(rotate=(-120, 120), mode='symmetric')),
            iaa.Sometimes(0.25, iaa.GaussianBlur(sigma=(0, 3.0))),

            # noise
            iaa.Sometimes(0.1,
                          iaa.OneOf([
                              iaa.Dropout(p=(0, 0.05)),
                              iaa.CoarseDropout(0.02, size_percent=0.25)
                          ])),

            iaa.Sometimes(0.25,
                          iaa.OneOf([
                              iaa.Add((-15, 15), per_channel=0.5), # brightness
                              iaa.AddToHueAndSaturation(value=(-10, 10), per_channel=True)
                          ])),

        ])
    def _define_task_file(self):
        if self.task == "oscc_bin":
            return self.oscc_bin_root
        if self.task  == "dys_bin":
            return self.dys_bin_root
        if self.task == "oscc_dys":
            return self.oscc_dys_root
        if self.task == "multiclass":
            return self.multiclass_root
        
    def _define_classes(self, task):
        if task == "oscc_bin":
            return {
                "oscc": 1,
                "not oscc": 0
            }
        if task  == "dys_bin":
            return {
                "dysplasia": 1,
                "no_dysplasia": 0
            }
        if task == "oscc_dys":
            return {
                "oscc": 1,
                "dysplasia": 0
            }
        if task == "multiclass":
            return {
                "oscc": 2,
                "dysplasia": 1,
                "no_dysplasia": 0
            }
        
    def data_per_fold(self, fold: int, train: bool = True):
        if train:
            data_table = self.data.train.copy()
            data_table["patch"] = data_table["patch"].apply(lambda x: self.patch_path / f"{x}.png" if str(x)[-3:] != "png" else self.patch_path / x).to_numpy()

            # returns 1 - (n-1) as train
            if fold == 0:
                fold = data_table["fold"].max()

            # invalid value for fold
            elif fold not in set(data_table["fold"].to_numpy()):
                raise ValueError("Number of folds out of range in csv file.")
            
            train_table = data_table[data_table["fold"] != fold].reset_index()
            test_table = data_table[data_table["fold"] == fold].reset_index()

            logger.info(f"Subset size: train {len(train_table)} - test {len(test_table)}")

            train_paths = train_table["patch"].to_numpy()
            test_paths  = test_table["patch"].to_numpy()

            train_labels = train_table["class"].to_numpy()
            test_labels  = test_table["class"].to_numpy()
            
            logger.info("Train set examples")
            logger.info(train_table[["patch", "class", "fold"]].sample(5))

            logger.info("Validation set examples")
            logger.info(test_table[["patch", "class", "fold"]].sample(5))

            return (train_paths, train_labels), (test_paths, test_labels)
        else:
            data_table = self.data.test.copy()
            test_paths = data_table["patch"].apply(lambda x: self.patch_path / f"{x}.png").to_numpy()
            test_labels = data_table["class"].to_numpy()

            logger.info("Test set examples")
            logger.info(data_table[["patch", "class"]].head(5))

            return (test_paths, test_labels)

    @property
    def origin_test(self):
        data_table = self.data.origin_test
        
        paths = data_table["path"].to_numpy()
        labels = data_table["class"].to_numpy()

        return paths, labels
