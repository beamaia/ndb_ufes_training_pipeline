import pandas as pd
import pytest

from src.dataset.leakage_safe_organizer import LeakageSafeNDBUfesOrganizer, MULTICLASS_LABELS
from src.validate_input.params import Parameters


def _params(tmp_path, csv_path):
    return Parameters.model_validate({
        "project": "multiclass",
        "run_name": "test",
        "stages": {"train": True, "test": True, "origin_test": False},
        "dataset": {
            "root": str(tmp_path),
            "patch": "patch_images",
            "origin": "original_images",
            "fold_assignments_path": str(csv_path),
            "cv_folds": [0, 1, 2, 3, 4],
            "test_fold": 5,
        },
        "training": {"mode": "finetune"},
        "early_stopping": {"enabled": True, "patience": 10, "min_delta": 0.001, "mode": "all"},
        "experiment": {"models": ["resnet50"]},
        "hyperparameters": {
            "optimizer": {"name": "sgd", "learning_rate": 0.01, "momentum": 0.9, "other": {}},
            "scheduler": {"name": "reduce_lr_on_plateau", "other": {"mode": "min"}},
            "model": {"name": "resnet50", "other": {}},
            "other": {
                "loss": "cross_entropy",
                "loss_weights": True,
                "folds": 5,
                "epochs": 1,
                "batch_size": 2,
                "train_type": "cross_validation",
            },
        },
        "device": "cpu",
        "node_type": "float32",
    })


def _fold_csv(tmp_path):
    rows = []
    labels = list(MULTICLASS_LABELS)
    for fold in range(6):
        for label_index, label in enumerate(labels):
            origin_id = fold * 10 + label_index
            rows.append({
                "origin_id": origin_id,
                "patch": f"p{fold}{label_index:03d}",
                "diagnosis": label,
                "fold": fold,
            })
    csv_path = tmp_path / "fold_assignments_patch_level.csv"
    pd.DataFrame(rows).to_csv(csv_path, index=False)
    return csv_path


def test_multiclass_label_mapping_is_stable():
    assert MULTICLASS_LABELS == {
        "Leukoplakia without dysplasia": 0,
        "Leukoplakia with dysplasia": 1,
        "OSCC": 2,
    }


def test_cv_split_never_uses_test_fold(tmp_path):
    csv_path = _fold_csv(tmp_path)
    organizer = LeakageSafeNDBUfesOrganizer(_params(tmp_path, csv_path))

    (train_paths, train_labels, train_metadata), (val_paths, val_labels, val_metadata) = organizer.data_per_fold(2, train=True)

    assert len(train_paths) == 12
    assert len(val_paths) == 3
    assert set(row["fold"] for row in train_metadata) == {0, 1, 3, 4}
    assert set(row["fold"] for row in val_metadata) == {2}
    assert 5 not in set(row["fold"] for row in train_metadata + val_metadata)
    assert set(train_labels).issubset({0, 1, 2})
    assert set(val_labels).issubset({0, 1, 2})


def test_heldout_test_fold_is_fold_five(tmp_path):
    csv_path = _fold_csv(tmp_path)
    organizer = LeakageSafeNDBUfesOrganizer(_params(tmp_path, csv_path))

    _, _, test_metadata = organizer.data_per_fold(5, train=False)

    assert len(test_metadata) == 3
    assert set(row["fold"] for row in test_metadata) == {5}


def test_origin_cannot_span_folds(tmp_path):
    csv_path = _fold_csv(tmp_path)
    table = pd.read_csv(csv_path)
    table.loc[1, "origin_id"] = table.loc[0, "origin_id"]
    table.loc[1, "fold"] = 1
    table.to_csv(csv_path, index=False)

    with pytest.raises(ValueError, match="multiple folds"):
        LeakageSafeNDBUfesOrganizer(_params(tmp_path, csv_path))
