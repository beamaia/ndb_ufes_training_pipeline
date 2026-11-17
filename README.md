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

Added after the article, all from public weights through timm (`TIMM_MODEL_IDS` in `src/models/model_selector.py` names the weights of each): `efficientnet_b0`, `efficientnet_b3`, `convnext_large`, `swin_tiny_patch4_window7_224`, `swin_base_patch4_window7_224`. The pathology foundation models are not built yet.

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
| `configs/trial.yaml` | A short run of one model (2 epochs) that goes to the experiment `ndb_ufes_trial`, never to `ndb_ufes_exp1` or `ndb_ufes_exp2`. |
| `configs/timing.yaml` | One epoch of every model, to see that each one trains on the machine and how long an epoch takes. It goes to `ndb_ufes_trial`. |
| `configs/runpod.yaml` | A RunPod machine: the CUDA device, the price per hour of one GPU, the MLflow address and the organizer location. |

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

## RunPod

A GPU with 48 GB (an A40 costs $0.59 an hour) trains every model at batch size 30. On the 16 GB of a laptop, VGG16, ConvNeXt-L and Swin-B do not fit well.

Everything that should survive a pod lives on a **network volume** (mounted at `/workspace`): this repository, the two files of the organizer, the patch images, the Python environment and the checkpoints. A volume belongs to one data center, and a pod can use it only in that data center, so create it where the GPU you want is available. The volume is billed per GB (the standard storage costs about half of the high-performance one), so keep it small: 30 GB holds the environment (about 8 GB), the patches (1.4 GB) and the download of the first run (about 8 GB, deleted afterwards), and leaves about 12 GB for checkpoints. The checkpoints of all the planned models come to tens of GB (an estimate), so copy each model's checkpoints to the laptop and delete them from the volume as you go, and delete the volume when the thesis runs are finished.

**Once**, on the first pod that has the volume attached:

1. Clone this repository in `/workspace`. Copy the two small files of the organizer into `/workspace/ndb_ufes_data_organizer/` (`runpodctl send`, `rsync`; 1 MB in all): `results/phase1/folds/fold_assignments.csv` and `results/phase0/dataset_tables/patch_table_public.csv`.
2. Write the `.env` in the repository yourself: `MLFLOW_TRACKING_URI=http://localhost:8002`, `MLFLOW_TRACKING_USERNAME`, `MLFLOW_TRACKING_PASSWORD`, `HF_TOKEN` when a gated model is used, and `MLFLOW_HTTP_REQUEST_MAX_RETRIES=10` with `MLFLOW_HTTP_REQUEST_BACKOFF_FACTOR=60`, so that a loss of the tunnel shorter than about 20 minutes does not end a run (MLflow allows at most 10 retries; a longer loss ends the run, and `--resume` continues it). Do not paste them in a chat or a commit. The password only works through your tunnel.

**Every time:**

1. From the laptop, with the MLflow stack running, open the tunnel that makes the laptop's MLflow (port 8000) appear as port 8002 on the pod (not 8001: the RunPod template runs nginx there), and leave it open: `ssh -N -o ExitOnForwardFailure=yes -o ServerAliveInterval=30 -R 8002:localhost:8000 -p <port> root@<ip>` (it stops with a message if the port is already taken on the pod, for example by an older tunnel, instead of failing silently). To keep it alive when the connection changes or drops, use `caffeinate -dimsu ./tunnel.sh <ip> <port>` instead: it frees the port that the old session holds on the pod and connects again, in a loop. The laptop has to stay awake (`caffeinate`).
2. On the pod, `git pull` if the code changed, then `./pod.sh --params configs/timing.yaml`. The script checks that `.env` and the two files are there, installs the environment (the first time only), downloads the patch images from the public Mendeley zip if they are missing (the first time only: about 3 GB and over an hour at the speed we measured, needing about 8 GB of free disk while it runs), trains, and **stops the pod** when it ends or fails. `KEEP_POD=1 ./pod.sh ...` leaves the pod running.

Use `configs/timing.yaml` first: one epoch of every model, to see that each one trains on the machine and how long an epoch takes. For the thesis runs use a file that extends `configs/runpod.yaml`, which also logs the cost of each run.

Several GPUs on one pod do not split one training. Each training stays on one GPU, and the models (or folds) are independent, so give each GPU its own settings file and start one process per GPU: start the first one and wait until it is training, so the environment and the images are in place, then start the others, for example `CUDA_VISIBLE_DEVICES=1 KEEP_POD=1 ./pod.sh --params configs/b.yaml`. Start all of them with `KEEP_POD=1`, because the first to finish would otherwise stop the pod under the others, and stop the pod yourself when all have ended (`runpodctl stop pod $RUNPOD_POD_ID`). The same seed gives the same result on the same kind of GPU, so the order does not matter. The images are loaded and augmented (imgaug) on the CPU, and with the models of this project that is slower than the GPU work, so the protocol (`params.yaml`) uses 8 worker processes (`run.num_workers`), for the 9 vCPUs of the pod; with the images loaded in the main process an epoch of MobileNetV2 took almost 4 minutes there.

## The official run

`configs/final.yaml` runs every model with the full protocol, the cheapest first, and stops starting new models once the run has cost `run.max_cost_usd` (the model that is running is not stopped, so keep the balance above it). A model that fails (out of memory, for example) is logged and the next one goes on; the failures are listed at the end.

Each fold keeps its best checkpoint and the checkpoint of its last epoch (`run.last_checkpoints: 1`). When the fold ends both are uploaded to MLflow (`run.checkpoints_in_mlflow: true`), which stores them in its artifacts folder on the laptop (`ndb_ufes_mlflow/var/mlflow/artifacts`) through the tunnel, so that if the credit ends or the pod dies only the fold that was training is lost. The upload happens inside the training, so the GPU waits for it. When a model has finished all its folds, its checkpoints are deleted from the pod (`run.delete_checkpoints_after_upload: true`), because the volume is small: the pod holds the checkpoints of one model at a time. A model that was deleted is not resumed by `--resume`: take the models that are done out of the list before you start again.

```bash
./pod.sh --params configs/final.yaml
```

## Path Foundation

Path Foundation (Google) is a TensorFlow model, so it has a script of its own, `path_foundation.py`, with its own dependencies in its header (`uv` builds a separate environment, so TensorFlow and PyTorch do not meet). It trains the model end to end with the protocol of the other models of Experiment 2: the same folds and test fold, the same augmentation, weighted cross-entropy, SGD at 0.001 with momentum 0.9, the learning rate reduced on a plateau, early stopping, a linear layer from the 384 numbers of the embedding to the three classes. It logs to MLflow with the names of the other runs, so `src/experiment_table.py` and the notebooks read it as they read the others. The settings are in `configs/path_foundation.yaml`.

```bash
uv run path_foundation.py --params configs/path_foundation.yaml
```

Two differences, both in the settings file or in the code and both there for a reason that was measured. The linear layer starts at zero (the embedding has a norm of about 48, and a random layer gave errors so large in the first steps that they damaged the model). And the gradient is clipped (`gradient_clip_norm`, 1.0): at the learning rate of the protocol, plain SGD made the model collapse to one class after the first epoch, and clipping, or a ten times smaller learning rate for the model, kept it learning. The model is downloaded from Hugging Face at a fixed commit and needs access to `google/path-foundation` in the account of the `HF_TOKEN`. The best weights of each fold stay in `models/path_foundation/` on the machine that trains it.

## Notebooks

`notebooks/exp_analysis.ipynb` asks whether the models keep making the same mistakes, from the saved test predictions of every fold run. `notebooks/exp2_explainability.ipynb` asks where the errors come from and what the models look at: the twelve models as a panel, the label-authority test (do the models follow the label of the patch or the diagnosis of its picture), error rates when the label agrees with the picture and when it does not, severity skipping, colour, overlap and leakage exposure, and Grad-CAM maps for the convolutional models. Both read MLflow on this computer and keep what they download in `notebooks/cache`. `uv run python -m src.validation_predictions --experiment ndb_ufes_exp2` makes the predictions of each fold model on its own validation fold, which the second notebook can use.

## Exporting the results

`uv run python -m src.experiment_table --export ../ndb_ufes_data_organizer/results/experiments` reads both experiments from MLflow (the server must be up) and writes a folder for each, `experiment_1` and `experiment_2`, with `summary.csv` and `summary.md` (one row for each model: balanced accuracy per patch, picture and patient, ROC AUC, recall of each class, epochs, minutes, cost), `fold_metrics.csv` (every test and training metric of every fold run) and `confusion_matrices.csv` (the cells added up over the folds, per model and level, with the share of the true class). `--experiment ndb_ufes_exp2` prints one summary table instead.

`uv run python -m src.panel --experiment ndb_ufes_exp2 --out ../ndb_ufes_data_organizer/results/experiments/experiment_2` adds the twelve models as a panel, from the saved test predictions (the same with `ndb_ufes_exp1` and `experiment_1` for the other experiment): `panel_patches.csv` (for each patch what each model says, how many of them are right, their consensus and mean severity, the label of the patch and the diagnosis of its picture), `panel_n_correct.csv` (how many patches are got right by 0 to 12 models, for each true class), `parent_vs_label.csv` (the patch labels against the diagnosis of their pictures, for all patches) and `label_authority.csv` (what the models say about the patches of each label, by the diagnosis of their parent picture, on the test fold and, when `src/validation_predictions.py` has been run, on the validation folds).

`uv run python -m src.model_comparison --results ../ndb_ufes_data_organizer/results/experiments/experiment_2` asks whether the models differ at all, from the `fold_metrics.csv` of the export (no MLflow needed), on the balanced accuracy of each fold run on the test fold. The main analysis is the one of the published article: Kruskal-Wallis over the models, then Conover-Iman for every pair with the Holm-Sidak correction, a critical-difference diagram, and CNNs against transformers with the fold scores of each family together (Mann-Whitney). Shapiro-Wilk and Levene are reported as in the article. Those tests treat the fold scores as independent, which they are not quite (the fold models share training data), so the paired Friedman test and the Nemenyi comparison are run as a check (Wilcoxon cannot reach p<0.05 with five folds). It writes `model_ranking.csv` (mean rank, mean and standard deviation of the balanced accuracy, family), `model_pairs.csv` (both p-values of every pair), `model_critical_difference.png` and `model_selection.md`, which gives the tests and the best CNN and best transformer with the models that cannot be told apart from each one.

`uv run python -m src.select_checkpoints --results ../ndb_ufes_data_organizer/results/experiments/experiment_2` picks, for VGG16, DenseNet-121 and Swin-T (`--models` changes them), the fold run with the highest validation balanced accuracy at the epoch of its kept checkpoint (`fold_metrics.csv` has it as `val_balanced_acc_at_best_epoch`; the test fold plays no part in the choice). It writes `representative_checkpoints.csv` and prints the `aws s3 cp` commands that bring just those checkpoints back from the archive of `ndb_ufes_mlflow`, to the folder where MLflow keeps them.

`uv run python -m src.occlusion --results ../ndb_ufes_data_organizer/results/experiments/experiment_2` draws occlusion maps for the three models of `representative_checkpoints.csv` (VGG16, DenseNet-121, Swin-T; `--models` changes them) on four groups of patches taken from `panel_patches.csv` (`--per-group` patches each, 30 by default, and at most `--per-picture` of one picture): labeled with dysplasia and called OSCC by the models like their OSCC picture (the models agree with the parent), the same with a picture that has dysplasia (a genuine model error), labeled with dysplasia and right for all twelve models, and wrong for all twelve. A window of a quarter of the patch (128 pixels on a patch of 512) covered with white slides in steps of an eighth (64 pixels), 7 x 7 positions, and the map is the drop of the probability of the class the model predicts. The checkpoints have to be on this computer (the commands of `src.select_checkpoints` bring them back from S3). It writes `occlusion_summary.csv` (for each patch and model: the prediction, the largest drop, how focal the map is and how much of it is in the middle of the patch), `occlusion_grids.csv` (the 7 x 7 values), `occlusion_agreement.csv` (the Spearman correlation between the maps of two models on the same patch), a figure of examples for each model and `occlusion_mean_maps.png`. It runs on the Mac GPU or the cpu, with 50 passes of the model for each patch.

`uv run python -m src.occlusion_overlap --results ../ndb_ufes_data_organizer/results/experiments/experiment_2` asks whether an occlusion map follows the tissue or the place in the patch. Patches of one picture overlap (97% of the patches have a neighbour that shares a quarter of their area, and half of them one that shares half of it), and every patch is an exact 512 x 512 crop at a known position (`x`, `y`, `x2`, `y2` of the patch table). For the patches of `src.occlusion` (same groups, same seed) it takes up to `--neighbours` patches of the same picture that share at least `--min-overlap` of their area, draws the maps of both and compares them in two ways: `content_rho` puts both in the coordinates of the picture and correlates them over the shared tissue (blocks of 32 pixels, Spearman), and `position_rho` correlates the two 7 x 7 maps in the coordinates of the patches. `random_position_rho` is the second for a patch and a random patch of another picture, which is what a pattern that is the same in every patch gives. A high `content_rho` means the same tissue matters wherever it sits in the patch; a high `position_rho` (above the random one) means the same place matters whatever tissue is there; both near 0 means the maps are not reproducible. It writes `occlusion_logit_overlap_pairs.csv` and `occlusion_logit_overlap.png` (`--score` and `--baseline` as in `src.occlusion`).

`uv run python -m src.occlusion_deletion --results ../ndb_ufes_data_organizer/results/experiments/experiment_2` checks whether the occlusion maps point at evidence the model uses (a deletion test). For the patches of `src.occlusion` it covers the k windows the map ranks most important, k random windows (10 draws) and the k least important, for k = 1, 3 and 5, and records the share of covered images whose predicted class changes and the mean drop of the log-odds of the original class. A faithful map changes the prediction much more when its top windows are covered than when random ones are. It writes `occlusion_logit_deletion.csv` (`--score` and `--baseline` as in `src.occlusion`).

## Playground

`uv run python -m src.playground --results ../ndb_ufes_data_organizer/results/experiments/experiment_2` opens a small site on this computer only (`http://127.0.0.1:8870/`; `--port` changes it, not to 8765), in the colours of the data wiki, to look at the models at work, not to make results. Four views: the test patches, filterable, with their true label and what VGG16, DenseNet-121 and Swin-T say (the stored test predictions, mean of the five fold runs, from `notebooks/cache/ndb_ufes_exp2.csv`); the picture of a patch with its test patches, where switching patches on and off recomputes the picture's call with the rule of the training (`results_summary.summarize_predictions`); the picture cut into a grid of square tiles (512 pixels by default), each painted with what the models say, to move through tile by tile; and a free crop drawn by hand. The tiles and free crops are classified live by the representative checkpoint of one fold run of each model (`representative_checkpoints.csv`, restored from S3). A crop that overlaps a labelled patch by at least 90% (intersection over union) shows that patch's true label; any other crop shows only what the models say and the patches it overlaps. A confidence threshold set at the top marks every call below it as uncertain, and can leave such patches out of the picture's call. The left and right arrow keys go to the previous and next patch or picture, and all four move from tile to tile in the grid. It reads the private patch table of the organizer for the SAB pictures, so the page never leaves this computer and the server never takes a file name from the page. Crops that are not square or not 512 pixels are stretched to the model's input, which the models were not trained on; a grid is limited to 600 tiles.

## Experiment 1

`configs/experiment_1.yaml` trains the same models with the same protocol on the folds published with P-NDB-UFES, which leak, for the comparison with Experiment 2 (`configs/final.yaml`, the new folds). It only changes the folds file and the MLflow experiment (`ndb_ufes_exp1`). The folds file is made by the organizer (`uv run python scripts/phase1.py published-folds`) and has the same patches and labels, the published fold of each patch (the sixth published fold is the test fold) and every patch as its own case group, which is how a file whose folds share patients passes the check that a case group is in one fold. Copy `results/phase1/folds/fold_assignments_published.csv` to the pod next to the other organizer files, then `./pod.sh --params configs/experiment_1.yaml`.

The only flags are `--params` (one or more files), `--download-data` (get the patch images that are missing from `dataset.source_zip`, the public Mendeley zip, and go on; nothing is downloaded when all the images are there), `--check-data` and `--resume` (continue folds that were interrupted). The data check (the folds file and the images) runs once for each dataset in a run, not once for each settings file. The device is `auto` by default: CUDA, then Apple MPS, then the CPU. Runs are logged to the MLflow experiment `ndb_ufes_<purpose>`; without a server, set `run.tracking_uri` to a SQLite file such as `sqlite:///mlflow.db`.

## What a run produces

- In MLflow: one parent run per model (named for example `resnet50_sgd`) and one nested run per fold (`resnet50_sgd_fold_0` to `resnet50_sgd_fold_4`, tagged with the model, its family and the validation fold), with the validation metrics, the held-out test metrics, the predictions with class probabilities, the classification report, the split manifest, and the mean and standard deviation over the folds. The metrics are grouped in sections on the board (a slash in a name makes a section): `train` and `val` (one point per epoch), `test_patch`, `test_picture`, `test_patient` and `test_picture_vs_ndb` (the held-out test, judged per patch, per picture, per patient, and per picture against its NDB-UFES diagnosis), each with its area under the ROC curve of each class (`roc_auc_<class>`, `roc_auc_macro`), and a `..._cm` section with the confusion matrix as numbers, one metric for each cell: `test_patch_cm/oscc_as_leukoplakia_with_dysplasia` counts the patches of the true class that the model put in the other class. `training` holds the learning rate, the seconds, the epochs and the memory, and `run` the seconds and the cost. The parent run has the same names, with one point for each fold and the mean and standard deviation of each metric as `<name>_avg` and `<name>_std`. The matrix is saved as `confusion_matrix.json` as well. The parameters, the git commit, the versions of the main packages and the hash of the folds file are logged too.
- On disk, in `models/project_multiclass_run_<run>/fold_<k>/`: for each fold, the best checkpoint (lowest validation loss) and the checkpoint of its last epoch (`run.last_checkpoints: 1`; older ones are deleted as training goes on, to save disk space; 0 keeps only the best), plus a tiny file with the results of the fold, which is what `--resume` reads. While a fold is training, a larger file with the weights and the optimizer lets an interrupted fold continue; it is reduced to the results when the fold ends. A copy of each checkpoint in MLflow is off by default (`run.checkpoints_in_mlflow`). A checkpoint describes itself: the weights and the model name, the classes, the preprocessing, the validation fold, the test fold, the seed, the hash of the folds file and the epoch. `src/checkpoint.py` rebuilds the model from the file alone.

## Reproducibility

Every fold starts by seeding Python, numpy, torch and the augmentation library with the seed plus the fold number, and sets the deterministic flags of torch. The same seed gives the same result on the same hardware. A different kind of hardware (for example a Mac and an NVIDIA GPU) can differ in the last digits, and on a GPU a few operations (the backward pass of the adaptive average pooling of VGG16 and others, for example) have no deterministic version in PyTorch, so even on the same GPU two runs agree closely but not always to the last digit. Each loader worker seeds its own augmenter from the seed of the fold, so the augmentations are the same on every run with the same `run.num_workers`, but a run with another number of workers draws other augmentations: keep the number fixed for the models that are compared. The tests check the seeding, the splits, the metrics, the early stopping and the checkpoints.

## Licence and citation

The code is licensed under GPL-3.0-only. Cite the software using `CITATION.cff`, and cite the NDB-UFES and P-NDB-UFES datasets separately. Some pathology foundation models are released under non-commercial licences; this code does not include their weights.
