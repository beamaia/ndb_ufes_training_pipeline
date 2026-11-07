"""Colour filters before inference: the best checkpoint of each model predicts the test fold again, with every patch changed by a fixed colour filter
before the usual transform (grayscale, one stain only, a common stain, or a shifted hue). Nothing is trained again. If a filter changes the predictions,
the models use what the filter takes away. The filters that remove colour give images the models never saw, so a drop there can also mean "out of
distribution"; the stain normalisation is the closest to the training images and the hue shifts show whether the damage grows with the change.

    uv run python -m src.color_filters --results ../ndb_ufes_data_organizer/results/experiments/experiment_2
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from skimage.color import hed2rgb, lab2rgb, rgb2hed, rgb2lab

from src import logger
from src.config import resolve_device
from src.occlusion import PATCHES, explain_on_device, transform_and_white
from src.panel import CLASSES, ORGANIZER, PROBABILITIES, confusion_by_zoom, patch_table
from src.select_checkpoints import MLFLOW

FOLDS = ORGANIZER / "results/phase1/folds/fold_assignments.csv"
TEST_FOLD = 5  # configs: test_fold
HUE_SHIFTS = (20, 40)  # degrees, both ways; larger than the hue change of the training augmentation
BOOTSTRAPS = 2000


def as_float(image: Image.Image) -> np.ndarray:
    return np.asarray(image.convert("RGB"), dtype=np.float64) / 255


def as_image(rgb: np.ndarray) -> Image.Image:
    return Image.fromarray((np.clip(rgb, 0, 1) * 255).round().astype(np.uint8))


def grayscale(rgb: np.ndarray) -> np.ndarray:
    """The luminance (ITU-R 601 weights) in all three channels: the structure stays, the colour goes."""
    return np.repeat((0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2])[..., None], 3, axis=2)


def one_stain(rgb: np.ndarray, keep: int) -> np.ndarray:
    """The colour deconvolution of H&E (Ruifrok and Johnston, as in scikit-image) with only one stain kept: 0 hematoxylin (purple), 1 eosin (pink)."""
    hed = rgb2hed(rgb)
    hed[..., [k for k in range(3) if k != keep]] = 0
    return hed2rgb(hed)


def reinhard(rgb: np.ndarray, target: tuple[np.ndarray, np.ndarray]) -> np.ndarray:
    """Reinhard stain normalisation: each channel of the CIELAB colour is moved to the mean and standard deviation of the target."""
    lab = rgb2lab(rgb)
    mean, sd = lab.reshape(-1, 3).mean(axis=0), lab.reshape(-1, 3).std(axis=0) + 1e-6
    return lab2rgb((lab - mean) / sd * target[1] + target[0])


def hue_shift(image: Image.Image, degrees: int) -> Image.Image:
    """The hue turned by ``degrees`` (PIL keeps hue in 0-255), saturation and value unchanged."""
    h, s, v = image.convert("HSV").split()
    h = h.point(lambda x: (x + round(degrees / 360 * 256)) % 256)
    return Image.merge("HSV", (h, s, v)).convert("RGB")


def lab_target(patch_ids: list[str]) -> tuple[np.ndarray, np.ndarray]:
    """The mean, over the given patches, of the CIELAB mean and standard deviation of each patch: the stain every patch is normalised to."""
    stats = np.array([[lab.mean(axis=0), lab.std(axis=0)] for lab in (rgb2lab(as_float(Image.open(PATCHES / f"{p}.png"))).reshape(-1, 3) for p in patch_ids)])
    return stats[:, 0].mean(axis=0), stats[:, 1].mean(axis=0)


def filters(target: tuple[np.ndarray, np.ndarray]) -> dict:
    """Each filter, a function from the patch image to the filtered image."""
    chosen = {
        "none": lambda image: image,
        "grayscale": lambda image: as_image(grayscale(as_float(image))),
        "hematoxylin only": lambda image: as_image(one_stain(as_float(image), 0)),
        "eosin only": lambda image: as_image(one_stain(as_float(image), 1)),
        "stain normalised": lambda image: as_image(reinhard(as_float(image), target)),
    }
    for degrees in HUE_SHIFTS:
        for sign in (1, -1):
            chosen[f"hue {sign * degrees:+d}°"] = lambda image, d=sign * degrees: hue_shift(image, d)
    return chosen


def predict(model_name: str, model: torch.nn.Module, preprocessing: dict, images: dict[str, Image.Image], device: str, *_unused, batch_size: int = 32) -> list[dict]:
    """The class probabilities of the model for every image (``explain_on_device`` passes two more arguments that prediction does not use)."""
    transform, _ = transform_and_white(preprocessing)
    model = model.to(device).eval()
    ids, rows = list(images), []
    with torch.no_grad():
        for start in range(0, len(ids), batch_size):
            batch = torch.stack([transform(images[i]) for i in ids[start:start + batch_size]]).to(device)
            probabilities = torch.softmax(model(batch).float(), dim=1).cpu().numpy()
            rows += [{"patch_id": i, **dict(zip(PROBABILITIES, p)), "y_pred": int(p.argmax())} for i, p in zip(ids[start:start + batch_size], probabilities)]
    return rows


def balanced_accuracy(frame: pd.DataFrame) -> float:
    return float(frame.assign(right=frame["y_true"] == frame["y_pred"]).groupby("y_true")["right"].mean().mean())


def summary(predictions: pd.DataFrame, seed: int = 0) -> pd.DataFrame:
    """For each model and filter (and all models together): balanced accuracy, its change from no filter, and the share of patches whose class changes.
    The interval of the change (95%, ``all models``) comes from resampling the pictures within each true class: the patches of a picture are not
    independent, so the picture is the unit that is drawn."""
    base = predictions[predictions["filter"] == "none"].set_index(["model", "patch_id"])["y_pred"]
    rows = []
    for (model, name), part in [*predictions.groupby(["model", "filter"]), *((("all models", f), p) for f, p in predictions.groupby("filter"))]:
        flips = (part.set_index(["model", "patch_id"])["y_pred"] != base.loc[part.set_index(["model", "patch_id"]).index]).mean()
        rows.append({"model": model, "filter": name, "balanced_acc": balanced_accuracy(part), "flip_rate": float(flips)})
    table = pd.DataFrame(rows)
    table["change"] = table["balanced_acc"] - table["model"].map(table[table["filter"] == "none"].set_index("model")["balanced_acc"])

    # bootstrap over pictures: right and total model calls of each picture under each filter
    calls = predictions.assign(right=predictions["y_true"] == predictions["y_pred"]).groupby(["filter", "histopath_id", "y_true"])["right"].agg(["sum", "size"])
    right, total = calls["sum"].unstack("filter").fillna(0), calls["size"].unstack("filter").fillna(0)
    true = right.index.get_level_values("y_true").to_numpy()
    rng, draws = np.random.default_rng(seed), []
    for _ in range(BOOTSTRAPS):
        pick = np.concatenate([rng.choice(np.flatnonzero(true == c), size=(true == c).sum()) for c in np.unique(true)])
        r, t, k = right.to_numpy()[pick], total.to_numpy()[pick], true[pick]
        draws.append(np.mean([r[k == c].sum(axis=0) / t[k == c].sum(axis=0) for c in np.unique(true)], axis=0))
    draws = np.array(draws) - np.array(draws)[:, [list(right.columns).index("none")]]
    low, high = np.percentile(draws, [2.5, 97.5], axis=0)
    interval = pd.DataFrame({"filter": right.columns, "change_low": low, "change_high": high, "model": "all models"})
    return table.merge(interval, on=["model", "filter"], how="left").round(4)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--results", type=Path, required=True, help="The folder of an experiment in results/experiments, with representative_checkpoints.csv; the tables go there.")
    parser.add_argument("--models", nargs="+", help="Default: every model of representative_checkpoints.csv.")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--artifacts", type=Path, default=MLFLOW / "var/mlflow/artifacts", help="Where MLflow keeps the checkpoints (restored from S3).")
    args = parser.parse_args()

    checkpoints = pd.read_csv(args.results / "representative_checkpoints.csv").set_index("model")
    models = args.models or list(checkpoints.index)
    paths = {model: args.artifacts / checkpoints.loc[model, "checkpoint"] for model in models}
    missing = [str(path) for path in paths.values() if not path.exists()]
    if missing:
        raise SystemExit("Checkpoints missing, bring them back from S3 with the commands of src.select_checkpoints:\n" + "\n".join(missing))

    folds = pd.read_csv(FOLDS).dropna(subset=["fold"])
    test, training = folds.loc[folds["fold"] == TEST_FOLD, "patch_id"].tolist(), folds.loc[folds["fold"] != TEST_FOLD, "patch_id"].tolist()
    target = lab_target(training)  # from the cross-validation folds only: the test fold plays no part in the filter
    logger.info(f"{len(test)} test patches; the stain target comes from {len(training)} patches of the other folds (CIELAB mean {target[0].round(1)}, sd {target[1].round(1)})")

    meta, device, rows = patch_table(), resolve_device(args.device), []
    for name, change in filters(target).items():  # the filtered patches of one filter are held in memory while every model predicts them
        images = {patch_id: change(Image.open(PATCHES / f"{patch_id}.png").convert("RGB")) for patch_id in test}
        for model, path in paths.items():
            predicted, _ = explain_on_device(model, path, images, device, None, None, work=predict)
            rows += [{"model": model, "filter": name, **row} for row in predicted]
        logger.info(f"{name}: {len(paths)} models")

    predictions = pd.DataFrame(rows)
    predictions["y_true"] = predictions["patch_id"].map(meta["label"]).map({c: k for k, c in enumerate(CLASSES)})
    predictions["histopath_id"] = predictions["patch_id"].map(meta["histopath_id"])
    predictions.round(5).to_csv(args.results / "color_filters_predictions.csv", index=False)

    table = summary(predictions)
    table.to_csv(args.results / "color_filters.csv", index=False)
    by_zoom = pd.concat({name: confusion_by_zoom(part, meta) for name, part in predictions.groupby("filter", sort=False)}, names=["filter"]).reset_index("filter")
    by_zoom.to_csv(args.results / "color_filters_by_zoom.csv", index=False)

    stored = pd.read_csv(Path(__file__).resolve().parents[1] / "notebooks/cache" / f"{args.results.name.replace('experiment_', 'ndb_ufes_exp')}.csv", keep_default_na=False)
    stored = stored[stored["run_id"].isin(checkpoints.loc[models, "run_id"])].set_index(["model", "patch_id"])["y_pred"]
    none = predictions[predictions["filter"] == "none"].set_index(["model", "patch_id"])["y_pred"]
    logger.info(f"Check: with no filter, {(none == stored.loc[none.index]).mean():.1%} of the predictions equal those the training stored for these runs")

    pooled = table[table["model"] == "all models"].set_index("filter")
    print(pooled[["balanced_acc", "change", "change_low", "change_high", "flip_rate"]].to_string())
    recall = by_zoom[(by_zoom["model"] == "all models") & (by_zoom["true"] == by_zoom["predicted"])]
    print("\nRecall % of each true class and zoom, all models:")
    print(recall.pivot_table(index="filter", columns=["true", "zoom"], values="share of true class %", sort=False).to_string())


if __name__ == "__main__":
    main()
