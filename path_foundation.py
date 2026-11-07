# /// script
# requires-python = ">=3.11,<3.14"
# dependencies = [
#     "tensorflow[and-cuda]>=2.20; sys_platform == 'linux'",
#     "tensorflow>=2.20; sys_platform != 'linux'",
#     "tf-keras",
#     "huggingface_hub",
#     "mlflow==3.17.1",
#     "imgaug",
#     "numpy",
#     "pandas",
#     "scikit-learn",
#     "pillow",
#     "python-dotenv",
#     "pyyaml",
# ]
# ///
"""Path Foundation (Google, TensorFlow) fine-tuned end to end on P-NDB-UFES, with the protocol of the other models of Experiment 2: the same folds (validation fold 0 to 4
in turn, fold 5 for the test), the same augmentation, weighted cross-entropy, SGD, the learning rate reduced on a plateau, early stopping, and a linear layer from the
384 numbers of the embedding to the three classes (started at zero). One difference: the gradient is clipped (see gradient_clip_norm in the settings), because plain SGD at the
learning rate of the protocol made the model collapse to one class after the first epoch. The results go to MLflow with the names of the other runs, so the results table and the notebooks read them as they read
the others. It is a file of its own because TensorFlow and PyTorch do not share an environment well: the settings are in a YAML file, and the few things that the PyTorch
code does (the augmentation, the class weights, the early stopper, the metrics) are written here again, in their simple form.

    uv run path_foundation.py --params configs/path_foundation.yaml
"""
import argparse
import os
import random
import sys
import time
from multiprocessing import get_context
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from dotenv import dotenv_values
from PIL import Image
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.utils.class_weight import compute_class_weight

if not hasattr(np, "sctypes"):  # imgaug still reads the NumPy 1.x table of types
    np.sctypes = {"int": [np.int8, np.int16, np.int32, np.int64], "uint": [np.uint8, np.uint16, np.uint32, np.uint64], "float": [np.float16, np.float32, np.float64],
                  "complex": [np.complex64, np.complex128], "others": [bool, object, str, bytes]}
from imgaug import augmenters as iaa

REPO = Path(__file__).resolve().parent
CLASSES = ["Leukoplakia without dysplasia", "Leukoplakia with dysplasia", "OSCC"]  # the class numbers 0, 1, 2
KEYS = ["leukoplakia_without_dysplasia", "leukoplakia_with_dysplasia", "oscc"]
SIZE = 224


def load_env() -> None:
    """The MLflow credentials and the Hugging Face token from the .env files; nothing else is read."""
    for file in (REPO / ".env", Path(os.getenv("MLFLOW_ENV_FILE", REPO.parent / "ndb_ufes_mlflow" / ".env"))):
        if file.exists():
            for key, value in dotenv_values(file).items():
                if value and (key.startswith("MLFLOW_") or key == "HF_TOKEN"):
                    os.environ.setdefault(key, value)


def augmenter() -> iaa.Sequential:
    """The augmentation of the other models (src/dataset/leakage_safe_organizer.py)."""
    return iaa.Sequential([
        iaa.Sometimes(0.25, iaa.Affine(scale={"x": (1.0, 2.0), "y": (1.0, 2.0)})),
        iaa.Resize({"height": SIZE, "width": SIZE}),
        iaa.Fliplr(0.5),
        iaa.Flipud(0.2),
        iaa.Sometimes(0.25, iaa.Affine(rotate=(-120, 120), mode="symmetric")),
        iaa.Sometimes(0.25, iaa.GaussianBlur(sigma=(0, 3.0))),
        iaa.Sometimes(0.1, iaa.OneOf([iaa.Dropout(p=(0, 0.05)), iaa.CoarseDropout(0.02, size_percent=0.25)])),
        iaa.Sometimes(0.25, iaa.OneOf([iaa.Add((-15, 15), per_channel=0.5), iaa.AddToHueAndSaturation(value=(-10, 10), per_channel=True)])),
    ])


_augmenter = None


def load_batch(job):
    """In a worker: the images of a batch as one uint8 array, augmented (training) or only resized, each batch with its own seed so that the result does not depend on the worker."""
    global _augmenter
    paths, train, seed = job
    _augmenter = _augmenter or augmenter()
    if train:
        _augmenter.seed_(seed)
    images = [Image.open(path).convert("RGB") for path in paths]
    return np.stack([_augmenter.augment_image(np.asarray(image)) if train else np.asarray(image.resize((SIZE, SIZE), Image.BILINEAR)) for image in images])


def batches(pool, paths, order, train: bool, seed: int, size: int):
    """(indices, images) for each batch, in order; the images are loaded by the workers of the pool while the model works."""
    chunks = [order[i:i + size] for i in range(0, len(order), size)]
    jobs = [([paths[j] for j in chunk], train, int(np.random.SeedSequence([seed, k]).generate_state(1)[0])) for k, chunk in enumerate(chunks)]
    return zip(chunks, pool.imap(load_batch, jobs) if pool else map(load_batch, jobs))


def read_table(settings: dict) -> pd.DataFrame:
    """The patches of the folds file, with the picture and the diagnosis of the picture from the patch table."""
    organizer = (REPO / settings["organizer"]).resolve()
    table = pd.read_csv(organizer / settings["folds_csv"], dtype=str, keep_default_na=False)
    table = table[table["fold"] != ""].copy()  # the exact duplicates have no fold
    table["fold"] = table["fold"].astype(int)
    pictures = pd.read_csv(organizer / settings["patch_table"], dtype=str, keep_default_na=False).set_index("patch_id")[["histopath_id", "ndb_diagnosis"]]
    table = table.join(pictures, on="patch_id")
    table["y_true"] = table["patch_diagnosis"].map({name: i for i, name in enumerate(CLASSES)})
    table["path"] = [str(organizer / settings["patch_dir"] / f"{patch_id}.png") for patch_id in table["patch_id"]]
    if settings.get("limit_per_fold"):  # to try the script on few patches
        table = pd.concat([group.sample(min(len(group), settings["limit_per_fold"]), random_state=0) for _, group in table.groupby("fold")])
    return table.reset_index(drop=True)


class EarlyStopper:
    """Stops when the validation loss has not improved by min_delta for patience epochs (the "all" mode of src/early_stopper.py)."""

    def __init__(self, patience: int, min_delta: float):
        self.patience, self.min_delta, self.counter, self.best = patience, min_delta, 0, float("inf")

    def should_stop(self, loss: float) -> bool:
        if loss < self.best:
            self.best, self.counter = loss, 0
        elif loss >= self.best + self.min_delta:
            self.counter += 1
        return self.counter >= self.patience


def scores(truth, predicted) -> dict:
    labels = [0, 1, 2]
    return {"acc": accuracy_score(truth, predicted), "balanced_acc": balanced_accuracy_score(truth, predicted),
            "precision": precision_score(truth, predicted, labels=labels, average="macro", zero_division=0), "recall": recall_score(truth, predicted, labels=labels, average="macro", zero_division=0),
            "macro_f1": f1_score(truth, predicted, labels=labels, average="macro", zero_division=0)}


def test_metrics(predictions: pd.DataFrame) -> dict:
    """The metrics of the test fold, with the names (sections of the MLflow board) of the other runs: per patch, and per picture and per patient from the mean probabilities."""
    probabilities = predictions[[f"prob_{key}" for key in KEYS]].to_numpy()
    truth, predicted = predictions["y_true"].to_numpy(), predictions["y_pred"].to_numpy()
    metrics = {f"test_patch/{name}": value for name, value in scores(truth, predicted).items()}
    for index, key in enumerate(KEYS):
        if (truth == index).any() and not (truth == index).all():
            metrics[f"test_patch/roc_auc_{key}"] = roc_auc_score(truth == index, probabilities[:, index])
    metrics["test_patch/roc_auc_macro"] = float(np.mean([v for k, v in metrics.items() if "roc_auc_" in k]))
    matrix = confusion_matrix(truth, predicted, labels=[0, 1, 2])
    metrics |= {f"test_patch_cm/{KEYS[i]}_as_{KEYS[j]}": float(matrix[i, j]) for i in range(3) for j in range(3)}
    for unit, column in (("picture", "histopath_id"), ("patient", "case_group")):
        grouped = predictions.groupby(column)
        mean_probabilities = grouped[[f"prob_{key}" for key in KEYS]].mean().to_numpy()
        majority = grouped["y_true"].agg(lambda s: int(2 - np.argmax(np.bincount(s, minlength=3)[::-1])))  # the most frequent label, a tie to the more severe
        metrics |= {f"test_{unit}/{name}": value for name, value in scores(majority.to_numpy(), mean_probabilities.argmax(axis=1)).items()}
    return metrics


def embedding(output):
    return output["output_0"] if isinstance(output, dict) else output


def run_fold(fold: int, table: pd.DataFrame, settings: dict, pool, model_dir: str, mlflow) -> dict:
    import tensorflow as tf
    import tf_keras as keras

    seed = settings["seed"] + fold
    random.seed(seed), np.random.seed(seed), tf.random.set_seed(seed)
    train = table[~table["fold"].isin([fold, settings["test_fold"]])].reset_index(drop=True)
    validation = table[table["fold"] == fold].reset_index(drop=True)
    test = table[table["fold"] == settings["test_fold"]].reset_index(drop=True)
    size = settings["batch_size"]

    backbone = keras.models.load_model(model_dir)  # the model of Google, loaded as a Keras model whose weights are trained
    head = keras.layers.Dense(3, kernel_initializer="zeros")  # the embedding is large (a norm of about 48): a random head gives huge errors at first, which damage the model; with zeros the first loss is the chance value
    head.build((None, 384))
    variables = backbone.trainable_variables + head.trainable_variables
    optimizer = keras.optimizers.SGD(learning_rate=settings["learning_rate"], momentum=settings["momentum"], global_clipnorm=settings.get("gradient_clip_norm"))  # the one difference from the other models: see the settings file
    optimizer.build(variables)
    present = np.unique(train["y_true"])
    weights = np.ones(3, dtype=np.float32)
    weights[present] = compute_class_weight("balanced", classes=present, y=train["y_true"])  # from the training set, as the other models
    class_weights = tf.constant(weights)

    def loss_of(logits, labels):  # the weighted cross-entropy, averaged over the weights of the batch as in PyTorch
        weights = tf.gather(class_weights, labels)
        return tf.reduce_sum(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels, logits=logits) * weights) / tf.reduce_sum(weights)

    @tf.function(reduce_retracing=True)
    def train_step(images, labels):
        with tf.GradientTape() as tape:
            logits = head(embedding(backbone(tf.cast(images, tf.float32) / 255.0, training=True)))
            loss = loss_of(logits, labels)
        optimizer.apply_gradients(zip(tape.gradient(loss, variables), variables))
        return loss, logits

    @tf.function(reduce_retracing=True)
    def predict_step(images):
        return head(embedding(backbone(tf.cast(images, tf.float32) / 255.0, training=False)))

    def evaluate(frame: pd.DataFrame):
        """The weighted loss (mean over the batches), and the probabilities of every patch."""
        losses, probabilities = [], np.zeros((len(frame), 3), dtype=np.float32)
        for chunk, images in batches(pool, frame["path"].tolist(), np.arange(len(frame)), False, 0, size):
            logits = predict_step(tf.constant(images))
            losses.append(float(loss_of(logits, tf.constant(frame["y_true"].to_numpy()[chunk]))))
            probabilities[chunk] = tf.nn.softmax(logits).numpy()
        return float(np.mean(losses)), probabilities

    stopper, started = EarlyStopper(settings["early_stopping"]["patience"], settings["early_stopping"]["min_delta"]), time.perf_counter()
    schedule, best_loss, best_weights, bad_epochs, schedule_best, learning_rate, epochs_run = settings["scheduler"], float("inf"), None, 0, float("inf"), settings["learning_rate"], 0
    for epoch in range(settings["epochs"]):
        epoch_started, order = time.perf_counter(), np.random.default_rng([seed, epoch]).permutation(len(train))
        truth, predicted, losses = [], [], []
        for chunk, images in batches(pool, train["path"].tolist(), order, True, int(np.random.SeedSequence([seed, epoch]).generate_state(1)[0]), size):
            labels = train["y_true"].to_numpy()[chunk]
            loss, logits = train_step(tf.constant(images), tf.constant(labels))
            losses.append(float(loss)), truth.extend(labels), predicted.extend(logits.numpy().argmax(axis=1))
        val_loss, probabilities = evaluate(validation)
        train_scores, val_scores = scores(truth, predicted), scores(validation["y_true"], probabilities.argmax(axis=1))
        mlflow.log_metrics({"train/loss": float(np.mean(losses)), **{f"train/{k}": v for k, v in train_scores.items()}, "val/loss": val_loss, **{f"val/{k}": v for k, v in val_scores.items()},
                            "training/learning_rate": learning_rate, "training/epoch_seconds": time.perf_counter() - epoch_started}, step=epoch)
        print(f"fold {fold} epoch {epoch + 1}: train loss {np.mean(losses):.3f}, validation loss {val_loss:.3f}, validation balanced accuracy {val_scores['balanced_acc'] * 100:.1f}%", flush=True)
        epochs_run = epoch + 1
        if val_loss < best_loss:
            best_loss, best_weights = val_loss, [v.numpy() for v in variables]
        if val_loss < schedule_best * (1 - 1e-4):  # ReduceLROnPlateau of PyTorch: a relative threshold of 1e-4
            schedule_best, bad_epochs = val_loss, 0
        else:
            bad_epochs += 1
            if bad_epochs > schedule["patience"]:
                learning_rate, bad_epochs = max(learning_rate * schedule["factor"], schedule["min_lr"]), 0
                optimizer.learning_rate.assign(learning_rate)
        if stopper.should_stop(val_loss):
            mlflow.log_metrics({"training/early_stopped_epoch": float(epoch + 1)})
            break

    for variable, weight in zip(variables, best_weights):  # the weights of the epoch with the lowest validation loss
        variable.assign(weight)
    kept = REPO / "models" / "path_foundation" / f"fold_{fold}"
    kept.mkdir(parents=True, exist_ok=True)
    np.savez(kept / "best_weights.npz", *best_weights)  # kept on this machine, like the checkpoints of the other models
    _, probabilities = evaluate(test)
    predictions = test.drop(columns=["path"]).assign(y_pred=probabilities.argmax(axis=1), stage="test", model="path_foundation", validation_fold=fold)
    for index, key in enumerate(KEYS):
        predictions[f"prob_{key}"] = probabilities[:, index]
    metrics = test_metrics(predictions)
    mlflow.log_metrics(metrics)
    mlflow.log_metrics({"training/epochs_run": float(epochs_run), "training/fold_seconds": time.perf_counter() - started})
    predictions.to_csv(kept / "predictions.csv", index=False)
    mlflow.log_artifact(str(kept / "predictions.csv"), "heldout_predictions")  # the same artifact name as the other runs
    print(f"fold {fold}: validation loss {best_loss:.3f}, test balanced accuracy {metrics['test_patch/balanced_acc'] * 100:.1f}%", flush=True)
    return {**metrics, **{f"val/{k}": v for k, v in val_scores.items()}}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--params", type=Path, default=REPO / "configs" / "path_foundation.yaml", help="The settings file.")
    args = parser.parse_args()
    settings = yaml.safe_load(args.params.read_text())
    load_env()
    table = read_table(settings)
    workers = settings["workers"]
    pool = get_context("fork" if sys.platform == "linux" else "spawn").Pool(workers) if workers else None  # before TensorFlow touches the GPU

    import mlflow
    from huggingface_hub import snapshot_download

    model_dir = snapshot_download(settings["huggingface"]["repo"], revision=settings["huggingface"]["revision"])  # needs the access to the model in the account of HF_TOKEN
    os.environ["TF_USE_LEGACY_KERAS"] = "1"
    mlflow.set_tracking_uri(settings["tracking_uri"] or os.getenv("MLFLOW_TRACKING_URI", "http://localhost:8000/"))
    experiment = mlflow.set_experiment(f"ndb_ufes_{settings['purpose']}")
    started, rows = time.perf_counter(), []
    tags = {"model": "path_foundation", "model_family": "pathology_foundation", "optimizer": "sgd", "task": "multiclass", "purpose": settings["purpose"], "test_fold": str(settings["test_fold"]),
            "seed": str(settings["seed"]), "epochs": str(settings["epochs"]), "batch_size": str(settings["batch_size"]), "host": "runpod" if os.getenv("RUNPOD_POD_ID") else "local"}
    with mlflow.start_run(run_name="path_foundation_sgd", experiment_id=experiment.experiment_id, tags=tags):
        mlflow.log_params({key: str(value) for key, value in settings.items()})
        mlflow.log_artifact(str(args.params), "config")
        for fold in [f for f in settings["folds"] if f != settings["test_fold"]]:
            with mlflow.start_run(nested=True, run_name=f"path_foundation_sgd_fold_{fold}", tags={"model": "path_foundation", "model_family": "pathology_foundation", "validation_fold": str(fold), "test_fold": str(settings["test_fold"])}):
                rows.append(run_fold(fold, table, settings, pool, model_dir, mlflow))
            mlflow.log_metrics(rows[-1], step=fold)  # the parent run: one point for each fold
        frame = pd.DataFrame(rows)
        mlflow.log_metrics({f"{name}_avg": float(frame[name].mean()) for name in frame} | {f"{name}_std": float(frame[name].std(ddof=0)) for name in frame})
        seconds = time.perf_counter() - started
        mlflow.log_metrics({"run/run_seconds": seconds} | ({"run/cost_usd": seconds / 3600 * settings["hourly_cost"]} if settings.get("hourly_cost") else {}))


if __name__ == "__main__":
    main()
