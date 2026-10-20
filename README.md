# ndb_ufes_training_pipeline

Training and evaluation of classification models on the P-NDB-UFES patches (oral squamous cell carcinoma, leukoplakia with dysplasia, leukoplakia without dysplasia). It reads the folds made by [ndb_ufes_data_organizer](../ndb_ufes_data_organizer), which keeps every patient, picture and group of related pictures in one fold.

The code is shared; the trained weights are not. Anyone with the same images and the same folds file can train the same models with the same seeds.

## Design

The six folds are used as in the original experiments, with only the folds changed:

- **Fold 5 is the test fold.** It is never used for training or validation.
- **Folds 0 to 4 are the cross-validation folds.** Each one is the validation fold in turn and the other four are the training set. That is five trainings per model.
- **Every fold model is tested on fold 5.** The results are the mean and the standard deviation over the five fold models (population standard deviation).
- A case group is in exactly one fold, and the program refuses a folds file that breaks this, that lacks a class in a fold, or that repeats a patch.

Training is end to end (all layers are trained from the ImageNet weights).

## Hyperparameters

They are the ones of the article, in [`params.yaml`](params.yaml):

| Setting | Value |
| --- | --- |
| Seed | 42 (each fold uses 42 plus the fold number) |
| Loss | Weighted cross-entropy (weights from the training set) |
| Optimizer | SGD, learning rate 0.001, momentum 0.9 |
| Batch size | 30 |
| Epochs | up to 150 |
| Scheduler | ReduceLROnPlateau on the validation loss: factor 0.1, patience 10, minimum learning rate 0.000001 |
| Early stopping | patience 15, minimum delta 0.001 |

The best checkpoint of a fold is the one with the lowest validation loss.

## Models

The seven models of the article: `resnet50`, `mobilenetv2`, `densenet121`, `vgg16` (convolutional networks) and `coat_lite_small`, `pit_s_distilled_224`, `vit_small_patch16_384` (transformers).

## Setup

```bash
UV_CACHE_DIR=.uv-cache uv sync --group dev
uv run python -m pytest -q
```

The data is not in this repository. It is read from the sibling repository `ndb_ufes_data_organizer` (the `dataset.organizer` setting in `params.yaml`): the folds file `results/phase1/folds/fold_assignments.csv`, the patch images in `data/ndb_ufes/patch_level/images`, and the patch table `results/phase0/dataset_tables/patch_table_public.csv`, which gives the picture of each patch.

The MLflow credentials (`MLFLOW_TRACKING_USERNAME`, `MLFLOW_TRACKING_PASSWORD`) and the Hugging Face token (`HF_TOKEN`) go in a `.env` file in this repository, which git ignores. If that file does not exist, the program reads the sibling `ndb_ufes_mlflow/.env`, or the file named in `MLFLOW_ENV_FILE`. Only the variables that start with `MLFLOW_` and `HF_TOKEN` are read.

## Settings

Every setting is in one YAML file, so that git versions it; the command line has no settings. `params.yaml` holds the protocol of the article and the defaults of a run. A file in `configs/` starts with `extends: ../params.yaml` and changes only the settings it names, so each run is described by one complete file. Give `--params` several files to run them one after the other, each as its own run (all the files are checked before the first run starts):

| File | What it changes |
| --- | --- |
| `params.yaml` | The protocol, the models, the data paths and the MLflow experiment (`final`). |
| `configs/trial.yaml` | A short run of one model (2 epochs) that goes to the experiment `ndb_ufes_trial`, never to `ndb_ufes_final`. |
| `configs/runpod.yaml` | A RunPod machine: the CUDA device, the price per hour, the MLflow address and the organizer location. |

To keep a setup of your own, copy one of them to a new file and commit it. Every run saves, in its MLflow artifacts under `config/`, the file it used, the files it extends, and `merged_params.yaml`, what they add up to.

## Run

Check the folds file and the images, then stop:

```bash
uv run python main.py --check-data
```

The thesis runs, with the settings of `params.yaml`:

```bash
uv run python main.py
```

A short trial:

```bash
uv run python main.py --params configs/trial.yaml
```

On RunPod:

```bash
uv run python main.py --params configs/runpod.yaml
```

The only flags are `--params` (one or more files), `--check-data` and `--resume` (continue folds that were interrupted). The data check (the folds file and the images) runs once for each dataset in a run, not once for each settings file. The device is `auto` by default: CUDA, then Apple MPS, then the CPU. Runs are logged to the MLflow experiment `ndb_ufes_<purpose>`; without a server, set `run.tracking_uri` to a SQLite file such as `sqlite:///mlflow.db`.

## What a run produces

- In MLflow: one parent run per model and one nested run per fold, with the validation metrics, the held-out test metrics (`heldout_test_*`), the predictions with class probabilities, the classification report, the split manifest, and the mean and standard deviation over the folds (`avg_*` and `std_*`). The metrics include the area under the ROC curve of each class, the metrics per picture and per patient (`heldout_test_picture_*`, `heldout_test_patient_*`), and the confusion matrix as numbers, one metric for each cell: `heldout_test_cm_<true class>_as_<predicted class>` counts the patches of the true class that the model put in the other class (for example `heldout_test_cm_oscc_as_leukoplakia_with_dysplasia`), also averaged over the folds. The matrix is saved as `confusion_matrix.json` as well. The parameters, the git commit, the versions of the main packages and the hash of the folds file are logged too.
- On disk, in `models/project_multiclass_run_<run>/fold_<k>/`: for each fold, the best checkpoint (lowest validation loss) and the checkpoint of its last epoch (`run.last_checkpoints: 1`; older ones are deleted as training goes on, to save disk space; 0 keeps only the best), plus a tiny file with the results of the fold, which is what `--resume` reads. While a fold is training, a larger file with the weights and the optimizer lets an interrupted fold continue; it is reduced to the results when the fold ends. A copy of each checkpoint in MLflow is off by default (`run.checkpoints_in_mlflow`). A checkpoint describes itself: the weights and the model name, the classes, the preprocessing, the validation fold, the test fold, the seed, the hash of the folds file and the epoch. `src/checkpoint.py` rebuilds the model from the file alone.

## Reproducibility

Every fold starts by seeding Python, numpy, torch and the augmentation library with the seed plus the fold number, and sets the deterministic flags of torch. The same seed gives the same result on the same hardware. A different kind of hardware (for example a Mac and an NVIDIA GPU) can differ in the last digits. The tests check the seeding, the splits, the metrics, the early stopping and the checkpoints.

## Licence and citation

The code is licensed under GPL-3.0-only. Cite the software using `CITATION.cff`, and cite the NDB-UFES and P-NDB-UFES datasets separately. Some pathology foundation models are released under non-commercial licences; this code does not include their weights.
