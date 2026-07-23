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
    "uni": {"size": 224, "mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225]},
    "virchow": {"size": 224, "mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225]},
    "ctranspath": {"size": 224, "mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225]},
    "mocov3_vit_small": {"size": 224, "mean": [0.5, 0.5, 0.5], "std": [0.5, 0.5, 0.5]},
    "vit_base_patch16_224": {"size": 224, "mean": [0.5, 0.5, 0.5], "std": [0.5, 0.5, 0.5]},
    "vit_large_patch16_224": {"size": 224, "mean": [0.5, 0.5, 0.5], "std": [0.5, 0.5, 0.5]},
    "vit_base_patch32_224": {
        "size": 224,
        "mean": [0.5, 0.5, 0.5],
        "std": [0.5, 0.5, 0.5],
    },
    "deit_base_patch16_224": {"size": 224, "mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225]},
    "swin_base_patch4_window7_224": {"size": 224, "mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225]},
    "efficientnet_b0": {"size": 224, "mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225]},
    "efficientnet_b1": {"size": 240, "mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225]},
    "efficientnetb4": {"size": 380, "mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225]},
    "coat_lite_small": {"size": 224, "mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225]},
    "pit_s_distilled_224": {"size": 224, "mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225]},
    "vit_small_patch16_384": {"size": 384, "mean": [0.5, 0.5, 0.5], "std": [0.5, 0.5, 0.5]},
}

DEFAULT_PREPROCESSING = {"size": 224, "mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225]}


class LeakageSafeNDBUfesOrganizer:
    def __init__(self, params):
        self.params = params
        self.root_path = pl.Path(params.dataset.root)
        self.patch_path = self.root_path / params.dataset.patch
        self.fold_assignments_path = pl.Path(params.dataset.fold_assignments_path)
        self.cv_folds = list(params.dataset.cv_folds)
        self.test_fold = params.dataset.test_fold
        self.group_column = params.dataset.group_column
        self.allow_origin_overlap = params.dataset.allow_origin_overlap
        self.allow_group_overlap = params.dataset.allow_group_overlap
        self.train_classes_dict = MULTICLASS_LABELS
        self.train_num_classes = len(self.train_classes_dict)
        self.node_type = getattr(torch, params.node_type)
        self.model_name = params.hyperparameters.model.name.value

        self.table = self._load_table()
        self._validate_table()
        self.aug = self._configure_aug(self._preprocessing["size"])
        self.train_transform = self._configure_transform(train=True)
        self.test_transform = self._configure_transform(train=False)

    @property
    def _preprocessing(self):
        return MODEL_PREPROCESSING.get(self.model_name, DEFAULT_PREPROCESSING)

    def _load_table(self):
        if not self.fold_assignments_path.exists():
            raise FileNotFoundError(
                f"Fold assignment CSV not found at {self.fold_assignments_path}. "
                "Run `dvc pull` or update dataset.fold_assignments_path."
            )
        table = pd.read_csv(self.fold_assignments_path)
        if "origin" in table.columns and "origin_id" not in table.columns:
            table = table.rename(columns={"origin": "origin_id"})
        if "image_name" in table.columns and "patch" not in table.columns:
            table = table.rename(columns={"image_name": "patch"})
        if "class" in table.columns and "diagnosis" not in table.columns:
            table = table.rename(columns={"class": "diagnosis"})
        required = {"origin_id", "patch", "diagnosis", "fold"}
        missing = required - set(table.columns)
        if missing:
            raise ValueError(f"Fold assignment CSV missing required columns: {sorted(missing)}")
        return table.copy()

    def _validate_table(self):
        required_values = ["origin_id", "patch", "diagnosis", "fold"]
        missing_values = {
            column: int(self.table[column].isna().sum())
            for column in required_values
            if self.table[column].isna().any()
        }
        if missing_values:
            raise ValueError(f"Required fold-assignment fields contain missing values: {missing_values}")

        duplicate_patches = self.table[self.table["patch"].duplicated(keep=False)]
        if not duplicate_patches.empty:
            raise ValueError(
                "Patch identifiers must be unique in a fold-assignment CSV; "
                f"duplicates include {duplicate_patches['patch'].head(10).tolist()}"
            )

        unknown_labels = set(self.table["diagnosis"].dropna()) - set(MULTICLASS_LABELS)
        if unknown_labels:
            raise ValueError(f"Unsupported diagnosis labels: {sorted(unknown_labels)}")

        expected_folds = set(self.cv_folds + [self.test_fold])
        try:
            fold_values = pd.to_numeric(self.table["fold"], errors="raise")
        except (TypeError, ValueError) as exc:
            raise ValueError("Fold assignments must contain integer values.") from exc
        if not np.all(fold_values == fold_values.astype(int)):
            raise ValueError("Fold assignments must contain integer values.")
        self.table["fold"] = fold_values.astype(int)
        actual_folds = set(self.table["fold"])
        if actual_folds != expected_folds:
            raise ValueError(f"Expected folds {sorted(expected_folds)}, found {sorted(actual_folds)}")

        origin_fold_counts = self.table.groupby("origin_id")["fold"].nunique()
        leaking_origins = origin_fold_counts[origin_fold_counts > 1]
        if not leaking_origins.empty and not self.allow_origin_overlap:
            raise ValueError(f"Origins assigned to multiple folds: {leaking_origins.index.tolist()[:10]}")

        if self.group_column in self.table.columns:
            group_table = self.table.dropna(subset=[self.group_column])
            group_fold_counts = group_table.groupby(self.group_column)["fold"].nunique()
            leaking_groups = group_fold_counts[group_fold_counts > 1]
            if not leaking_groups.empty and not self.allow_group_overlap:
                raise ValueError(
                    f"Groups in '{self.group_column}' assigned to multiple folds: "
                    f"{leaking_groups.index.tolist()[:10]}"
                )

        class_counts = self.table.groupby("fold")["diagnosis"].nunique()
        incomplete_folds = class_counts[class_counts < self.train_num_classes]
        if not incomplete_folds.empty:
            raise ValueError(
                "Every fold must contain all classes for weighted cross-entropy; "
                f"incomplete folds: {incomplete_folds.to_dict()}"
            )

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
        prep = self._preprocessing
        return transforms.Compose([
            transforms.Resize((prep["size"], prep["size"])),
            transforms.ToTensor(),
            transforms.ConvertImageDtype(self.node_type),
            transforms.Normalize(mean=prep["mean"], std=prep["std"]),
        ])

    def _table_to_dataset_arrays(self, table):
        prepared = table.copy().reset_index(drop=True)
        prepared["class"] = prepared["diagnosis"].map(MULTICLASS_LABELS).astype(int)
        prepared["patch_path"] = prepared.apply(self._row_to_patch_path, axis=1)
        paths = prepared["patch_path"].to_numpy()
        labels = prepared["class"].to_numpy()
        metadata = prepared.drop(columns=["patch_path"]).to_dict("records")
        return paths, labels, metadata

    def _image_path_to_path(self, path):
        path = pl.Path(str(path))
        return path if path.is_absolute() else self.root_path / path

    def _row_to_patch_path(self, row):
        if "image_path" in row and pd.notna(row["image_path"]):
            path = self._image_path_to_path(row["image_path"])
            if path.exists():
                return path
        return self._patch_to_path(row["patch"])

    def _patch_to_path(self, patch):
        patch = str(patch)
        filename = patch if patch.endswith(".png") else f"{patch}.png"
        return self.patch_path / filename

    def data_per_fold(self, fold: int, train: bool = True):
        if train:
            if fold not in self.cv_folds:
                raise ValueError(f"Validation fold must be one of {self.cv_folds}; got {fold}")
            train_table = self.table[self.table["fold"].isin([f for f in self.cv_folds if f != fold])]
            val_table = self.table[self.table["fold"] == fold]
            if self.test_fold in set(train_table["fold"]) or self.test_fold in set(val_table["fold"]):
                raise ValueError("Held-out test fold leaked into train/validation split.")
            logger.info(f"Subset size: train {len(train_table)} - validation {len(val_table)}")
            return self._table_to_dataset_arrays(train_table), self._table_to_dataset_arrays(val_table)

        test_table = self.table[self.table["fold"] == self.test_fold]
        logger.info(f"Held-out test subset size: {len(test_table)}")
        return self._table_to_dataset_arrays(test_table)

    def split_manifest(self, fold):
        train_folds = [f for f in self.cv_folds if f != fold]
        return {
            "train_folds": train_folds,
            "validation_fold": fold,
            "test_fold": self.test_fold,
            "train_rows": int(self.table[self.table["fold"].isin(train_folds)].shape[0]),
            "validation_rows": int(self.table[self.table["fold"] == fold].shape[0]),
            "test_rows": int(self.table[self.table["fold"] == self.test_fold].shape[0]),
        }

    def provenance(self):
        digest = hashlib.sha256(self.fold_assignments_path.read_bytes()).hexdigest()
        return {
            "fold_assignments_path": str(self.fold_assignments_path),
            "fold_assignments_sha256": digest,
            "row_count": int(len(self.table)),
            "fold_counts": {str(k): int(v) for k, v in self.table["fold"].value_counts().sort_index().items()},
            "class_counts": {k: int(v) for k, v in self.table["diagnosis"].value_counts().items()},
        }
