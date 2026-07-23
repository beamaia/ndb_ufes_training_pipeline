# ndb_ufes_training_pipeline

Training and evaluation pipeline for the three-class P-NDB-UFES patch
diagnosis task.

## v1.0.0 Experiment Contract

The thesis comparison contains exactly:

- Experiment 1 / Batch 1: the original-comparable patch split.
- Experiment 2 / Batch 2: the patient-first grouped,
  lower-contamination-risk split.
- MobileNetV2, DenseNet-121, and ResNet-50 with ImageNet initialization and
  full fine-tuning.

Folds `0..4` rotate as validation folds. For each rotation, the other four
folds are used for training and fold `5` remains held out. The five resulting
checkpoints are all evaluated on fold `5`; their held-out metrics are
summarized as mean and population standard deviation.

Batch 3 is archived exploratory work. It is not part of the final thesis model
comparison and requires the explicit `--allow-exploratory-batch3` flag.

## Frozen Hyperparameters

| Setting | Value |
| --- | --- |
| Seed | 42 |
| Loss | Weighted cross-entropy |
| Optimizer | SGD, momentum 0.9 |
| Initial learning rate | 0.001 |
| Batch size | 30 |
| Epoch ceiling | 150 |
| Scheduler | ReduceLROnPlateau on validation loss |
| Scheduler factor / patience / minimum LR | 0.1 / 10 / 0.000001 |
| Early-stopping patience / minimum delta | 15 / 0.001 |

Scheduler patience means ten validation-loss epochs without sufficient
improvement; it is not an unconditional reduction every ten epochs.

## Setup And Validation

```bash
UV_CACHE_DIR=.uv-cache uv sync --group dev
./train.sh --check-env
./train.sh --no-pull --check-data
UV_CACHE_DIR=.uv-cache uv run --group dev python -m pytest
```

`train.sh` pulls DVC data by default. Use `--no-pull` when the local checkout
is already current. MLflow credentials are loaded from `MLFLOW_ENV_FILE` when
set, or from the sibling `ndb_ufes_mlflow/.env` when present. No user-specific
absolute path is required.

Run the canonical queue:

```bash
DEVICE=auto bash scripts/start_pndb_ufes_queue.sh
```

The default queue enumerates two batches × three CNNs. See
[`scripts/README.md`](scripts/README.md) for smoke tests, explicit overrides,
and the archived exploratory launcher.

Run DVC through the repository wrapper:

```bash
./run_dvc.sh pull
./run_dvc.sh status
```

The wrapper removes stale virtual-environment paths and uses the repository's
uv-managed environment.

## Provenance Boundary

The public v1.0.0 results freeze the completed historical runs without
retraining. Some runs were produced from dirty working trees and cannot all be
attributed to one clean commit. The organizer release records the exact
dataset hashes, MLflow parent/child run IDs, stored metrics, and this
limitation.

The code is licensed under GPL-3.0-only. Cite the software using
`CITATION.cff`; cite NDB-UFES/P-NDB-UFES source datasets separately.
