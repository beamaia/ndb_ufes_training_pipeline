# ndb_ufes_training_pipeline

Leakage-safe training pipeline for the three-class NDB-UFES patch diagnosis task.

The current training contract is:

- train only the multiclass patch diagnosis labels;
- use the repo-local DVC-pulled `data/ndb_ufes/patch_level/csvs/fold_assignments_patch_level_with_images.csv`;
- run cross-validation on folds `0..4`;
- keep fold `5` as a held-out test fold;
- evaluate every CV checkpoint on fold `5`;
- log parameters, checkpoints, predictions, split manifests, dataset provenance, and aggregate metrics to MLflow.

Run experiments with:

```bash
./train.sh
```

Useful preflight checks:

```bash
./train.sh --check-env
./train.sh --no-pull --check-data
```

`train.sh` pulls DVC data by default before training. Use `./train.sh --no-pull` when the local data checkout is already current. The script loads MLflow credentials from `MLFLOW_ENV_FILE` when set, or from `/Volumes/ssd/thesis_organization/ndb_ufes_mlflow/.env` when that file exists. It also uses `.dvc/tmp/site_cache` for DVC state so DVC does not depend on a user-level cache directory.

The default sweep keeps the article-compatible `200` epoch ceiling and uses early stopping from `params.yaml`:

```yaml
early_stopping:
  enabled: true
  patience: 10
  min_delta: 0.001
  mode: all
```

Run tests with:

```bash
UV_CACHE_DIR=.uv-cache uv run --group dev pytest
```

Run DVC through the repo wrapper when your shell has an active virtualenv:

```bash
./run_dvc.sh pull
./run_dvc.sh status
```

The wrapper strips stale `VIRTUAL_ENV` values from other checkouts and runs DVC through this repo's uv-managed Python environment.

If you call uv directly in zsh, quote extras such as `uv add 'dvc[s3]'`; otherwise zsh treats `[s3]` as a filename glob. This project pins DVC S3 support in `pyproject.toml`, including `boto3`.
