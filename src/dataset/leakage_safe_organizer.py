import hashlib
import pathlib as pl

import pandas as pd
import numpy as np
import torch
from torchvision import transforms

# imgaug still reads the NumPy 1.x ``sctypes`` compatibility table.
if not hasattr(np, "sctypes"):
    np.sctypes = {
        "int": [np.int8, np.int16, np.int32, np.int64],
        "uint": [np.uint8, np.uint16, np.uint32, np.uint64],
        "float": [np.float16, np.float32, np.float64],
        "complex": [np.complex64, np.complex128],
        "others": [bool, object, str, bytes],
    }

from imgaug import augmenters as iaa

from src import logger


MULTICLASS_LABELS = {
    "Leukoplakia without dysplasia": 0,
    "Leukoplakia with dysplasia": 1,
    "OSCC": 2,
}


MODEL_PREPROCESSING = {
    "efficientnet_b0": {"size": 224, "mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225]},
    "efficientnet_b3": {"size": 288, "mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225]},
    "efficientnetb4": {"size": 380, "mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225]},
    "convnext_large": {"size": 224, "mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225]},
    "swin_tiny_patch4_window7_224": {"size": 224, "mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225]},
    "swin_base_patch4_window7_224": {"size": 224, "mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225]},
    "coat_lite_small": {"size": 224, "mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225]},
    "pit_s_distilled_224": {"size": 224, "mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225]},
    "vit_small_patch16_384": {"size": 384, "mean": [0.5, 0.5, 0.5], "std": [0.5, 0.5, 0.5]},
}

DEFAULT_PREPROCESSING = {"size": 224, "mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225]}


REQUIRED_COLUMNS = ["patch_id", "case_group", "fold", "patch_diagnosis"]


class LeakageSafeNDBUfesOrganizer:
    """The patches of ``fold_assignments.csv`` (made by the data organizer), split by fold.

    One fold is kept as the test set. The others are the cross-validation folds: each one is the validation set in turn and
    the rest are the training set. A case group is in exactly one fold, so no patient is on two sides.
    """

    def __init__(self, params, check: bool = True):
        """``check=False`` skips the validation of the folds file, for data that has already been checked (see ``src/data_check.py``)."""
        self.params = params
        self.patch_path = pl.Path(params.dataset.patch_dir)
        self.folds_csv = pl.Path(params.dataset.folds_csv)
        self.patch_table = pl.Path(params.dataset.patch_table) if params.dataset.patch_table else None
        self.folds = list(params.dataset.folds)
        self.test_fold = params.dataset.test_fold
        self.cv_folds = [fold for fold in self.folds if fold != self.test_fold]
        self.train_classes_dict = MULTICLASS_LABELS
        self.train_num_classes = len(self.train_classes_dict)
        self.node_type = getattr(torch, params.node_type)
        self.model_name = params.hyperparameters.model.name.value

        self.table = self._load_table()
        if check:
            self._validate_table()
        self.aug = self._configure_aug(self.preprocessing["size"])
        self.train_transform = self._configure_transform(train=True)
        self.test_transform = self._configure_transform(train=False)

    @property
    def preprocessing(self):
        return MODEL_PREPROCESSING.get(self.model_name, DEFAULT_PREPROCESSING)

    def _load_table(self):
        if not self.folds_csv.exists():
            raise FileNotFoundError(f"Folds file not found at {self.folds_csv}. Set dataset.organizer or dataset.folds_csv in the settings file.")
        table = pd.read_csv(self.folds_csv, dtype=str, keep_default_na=False)
        missing = [column for column in REQUIRED_COLUMNS if column not in table.columns]
        if missing:
            raise ValueError(f"The folds file is missing the columns {missing}")
        unfolded = table["fold"] == ""
        if unfolded.any():  # the only patches without a fold are exact duplicates of a patch that has one
            if "dropped_duplicate_of" not in table.columns or (table.loc[unfolded, "dropped_duplicate_of"] == "").any():
                raise ValueError("A patch has no fold and is not marked as a dropped duplicate")
            table = table[~unfolded]
        table["fold"] = table["fold"].astype(int)
        if self.patch_table is not None:  # the picture of each patch and the NDB-UFES diagnosis of that picture
            pictures = pd.read_csv(self.patch_table, dtype=str, keep_default_na=False).set_index("patch_id")[["histopath_id", "ndb_diagnosis"]]
            table = table.join(pictures, on="patch_id")
            if table["histopath_id"].isna().any():
                raise ValueError(f"{int(table['histopath_id'].isna().sum())} patches of the folds file are not in {self.patch_table}")
        return table.reset_index(drop=True)

    def _validate_table(self):
        if self.table["patch_id"].duplicated().any():
            raise ValueError(f"Patch ids repeat in the folds file, for example {self.table['patch_id'][self.table['patch_id'].duplicated()].head(5).tolist()}")
        unknown_labels = set(self.table["patch_diagnosis"]) - set(MULTICLASS_LABELS)
        if unknown_labels:
            raise ValueError(f"Unsupported diagnosis labels: {sorted(unknown_labels)}")
        if set(self.table["fold"]) != set(self.folds):
            raise ValueError(f"Expected the folds {sorted(self.folds)}, found {sorted(set(self.table['fold']))}")
        in_several = self.table.groupby("case_group")["fold"].nunique()
        if (in_several > 1).any():
            raise ValueError(f"Case groups in more than one fold: {in_several[in_several > 1].index.tolist()[:10]}")
        classes_per_fold = self.table.groupby("fold")["patch_diagnosis"].nunique()
        if (classes_per_fold < self.train_num_classes).any():
            raise ValueError(f"Every fold must hold all classes (weighted cross-entropy needs them); incomplete: {classes_per_fold[classes_per_fold < self.train_num_classes].to_dict()}")

    def _configure_aug(self, size):
        return iaa.Sequential([
            iaa.Sometimes(0.25, iaa.Affine(scale={"x": (1.0, 2.0), "y": (1.0, 2.0)})),
            iaa.Resize({"height": size, "width": size}),
            iaa.Fliplr(0.5),
            iaa.Flipud(0.2),
            iaa.Sometimes(0.25, iaa.Affine(rotate=(-120, 120), mode="symmetric")),
            iaa.Sometimes(0.25, iaa.GaussianBlur(sigma=(0, 3.0))),
            iaa.Sometimes(0.1, iaa.OneOf([
                iaa.Dropout(p=(0, 0.05)),
                iaa.CoarseDropout(0.02, size_percent=0.25),
            ])),
            iaa.Sometimes(0.25, iaa.OneOf([
                iaa.Add((-15, 15), per_channel=0.5),
                iaa.AddToHueAndSaturation(value=(-10, 10), per_channel=True),
            ])),
        ])

    def _configure_transform(self, train):
        prep = self.preprocessing
        return transforms.Compose([
            transforms.Resize((prep["size"], prep["size"])),
            transforms.ToTensor(),
            transforms.ConvertImageDtype(self.node_type),
            transforms.Normalize(mean=prep["mean"], std=prep["std"]),
        ])

    def _arrays(self, table):
        """Image paths, labels (0 to 2) and the table rows as records, for a Dataset."""
        paths = (self.patch_path / (table["patch_id"] + ".png")).to_numpy()
        labels = table["patch_diagnosis"].map(MULTICLASS_LABELS).to_numpy()
        return paths, labels, table.to_dict("records")

    def split(self, validation_fold):
        """(paths, labels, records) of the training, validation and test patches when ``validation_fold`` is the validation fold."""
        if validation_fold not in self.cv_folds:
            raise ValueError(f"The validation fold must be one of {self.cv_folds}; got {validation_fold}")
        train_folds = [fold for fold in self.cv_folds if fold != validation_fold]
        parts = [self.table[self.table["fold"].isin(folds)] for folds in (train_folds, [validation_fold], [self.test_fold])]
        groups = [set(part["case_group"]) for part in parts]
        if groups[0] & groups[1] or groups[0] & groups[2] or groups[1] & groups[2]:
            raise ValueError("A case group is on two sides of the split")  # cannot happen after the check at loading; it guards the split itself
        logger.info(f"Validation fold {validation_fold}: train {len(parts[0])} - validation {len(parts[1])} - test {len(parts[2])}")
        return tuple(self._arrays(part) for part in parts)

    def missing_images(self):
        """The patches whose image is not on disk."""
        return [patch_id for patch_id in self.table["patch_id"] if not (self.patch_path / f"{patch_id}.png").exists()]

    def split_manifest(self, validation_fold):
        rows = self.table["fold"].value_counts()
        train_folds = [fold for fold in self.cv_folds if fold != validation_fold]
        return {
            "train_folds": train_folds,
            "validation_fold": validation_fold,
            "test_fold": self.test_fold,
            "train_rows": int(sum(rows[fold] for fold in train_folds)),
            "validation_rows": int(rows[validation_fold]),
            "test_rows": int(rows[self.test_fold]),
        }

    def provenance(self):
        return {
            "folds_csv": str(self.folds_csv),
            "folds_csv_sha256": hashlib.sha256(self.folds_csv.read_bytes()).hexdigest(),
            "row_count": int(len(self.table)),
            "case_groups": int(self.table["case_group"].nunique()),
            "fold_counts": {str(k): int(v) for k, v in self.table["fold"].value_counts().sort_index().items()},
            "class_counts": {k: int(v) for k, v in self.table["patch_diagnosis"].value_counts().items()},
        }
