# P-NDB-UFES Batch Training

These scripts run the two canonical P-NDB-UFES thesis experiments and keep the
queue PID and logs under the training repository.

Set these paths before running if the repositories were moved:

```bash
export NDB_UFES_TRAINING_REPO="/path/to/ndb_ufes_training_pipeline"
export NDB_UFES_THESIS_REPO="/path/to/ndb_ufes_data_organizer"
export NDB_UFES_ORGANIZER_DATA="/path/to/ndb_ufes_data_organizer/data"
export NDB_UFES_MLFLOW_ENV="/path/to/ndb_ufes_mlflow/.env"
```

Stop a previous queue safely, including its child processes:

```bash
bash scripts/stop_pndb_ufes_queue.sh
```

Check MPS from a normal macOS Terminal session (outside the Codex sandbox):

```bash
"$NDB_UFES_TRAINING_REPO/.venv/bin/python" \
  "$NDB_UFES_TRAINING_REPO/scripts/check_mps.py"
```

The project launcher uses `--device auto`: it selects MPS when this check
passes and otherwise reports the CPU fallback explicitly.

After the device-level check passes, verify each model independently:

```bash
"$NDB_UFES_TRAINING_REPO/.venv/bin/python" \
  "$NDB_UFES_TRAINING_REPO/scripts/check_models_mps.py"
```

This probe reports model loading, MPS forward/backward compatibility, and the
CPU validation round-trip used by the VGG16 path.

The default queue contains MobileNetV2, DenseNet-121, and ResNet-50 and runs
Batch 1 followed by Batch 2. To select a subset explicitly, provide a
space-separated `MODELS` value or pass `--models`.

You can also double-click `scripts/run_mps_validation.command` in Finder. It
opens a normal Terminal session and saves the complete device/model report to
`logs/pndb_ufes_batches/mps_validation_external.log`.

Run a one-epoch smoke test:

```bash
"$NDB_UFES_TRAINING_REPO/.venv/bin/python" \
  "$NDB_UFES_TRAINING_REPO/scripts/run_pndb_ufes_batch.py" \
  --batch batch1 --models densenet121 --epochs 1 \
  --batch-size 30 --device mps --node-type float32 --run-suffix smoke
```

Start the complete queue after the smoke test appears in MLflow:

```bash
EPOCHS=150 BATCH_SIZE=30 DEVICE=mps NODE_TYPE=float32 \
  bash scripts/start_pndb_ufes_queue.sh
```

When `DEVICE=mps`, the launcher runs the device and model preflight first and
does not create the background queue if either check fails. To bypass only
after a successful manual check, set `NDB_UFES_MPS_PREFLIGHT=0`.

Follow the queue with:

```bash
tail -F logs/pndb_ufes_batches/queue_stdout.log
```

## Exploratory Batch 3

Batch 3 is preserved for historical inspection but is not part of the v1.0.0
thesis comparison. Run it only through
`scripts/exploratory/run_three_batch_legacy.sh` or pass
`--allow-exploratory-batch3` explicitly.
