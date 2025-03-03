from abc import ABC
import pathlib as pl

import pandas as pd

from torchvision import transforms

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
            
    def __init__(self, patch: str, origin: str, root: str, task: str):
        self.root_path = pl.Path(root)
        self.patch_path = pl.Path(self.root_path / patch)
        self.origin_path = pl.Path(self.root_path / origin)
        self.task = task
        
        self.folds_division_path = self._define_task_file()
        self.train_classes_dict = self._define_classes(self.task)
        self.train_num_classes = len(self.train_classes_dict)

        self.data = self.DataTable(self.root_path, self.folds_division_path)
        self.origin_classes_dict = self._define_classes("multiclass")

        self.train_transform = transforms.Compose([transforms.Resize((224, 224)),
                                                   transforms.ToTensor(),
                                                   transforms.RandomHorizontalFlip(0.5),
                                                   transforms.RandomVerticalFlip(0.5),
                                                   transforms.RandomApply([transforms.RandomRotation(10)], 0.25),
                                                   transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

        self.test_transform = transforms.Compose([transforms.Resize((224, 224)),
                                                  transforms.ToTensor()])

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
