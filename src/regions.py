import argparse
from functools import partial
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image

from src import logger
from src.config import resolve_device
from src.occlusion import explain_on_device, transform_and_white
from src.panel import CLASSES, ORGANIZER, load_predictions, prepare
from src.select_checkpoints import MLFLOW, MODELS

LAB_TABLE = ORGANIZER / "results/phase0/dataset_tables/private_lab/patch_table_lab.csv"  # private: the path of each picture; nothing of it is written out
CELL = 16  # coverage is measured on cells of this many pixels, counting only cells that a whole patch covers completely
SIZES = (640, 768, 1024)
OPTIONS = ("unmasked", "masked")


def windows(boxes: pd.DataFrame, width: int, height: int, size: int, step: int, floor: float) -> pd.DataFrame:
    """Square windows of a picture made of whole patches: the patches lying completely inside the window (at least two) cover at least ``floor`` of its
    area. ``boxes`` has the patches of the picture (index patch_id; columns x, y, x2, y2)."""
    x, y, x2, y2 = (boxes[column].to_numpy() for column in ("x", "y", "x2", "y2"))
    rows = []
    for y0 in range(0, height - size + 1, step):
        for x0 in range(0, width - size + 1, step):
            inside = (x >= x0) & (y >= y0) & (x2 <= x0 + size) & (y2 <= y0 + size)
            if inside.sum() < 2:
                continue
            mask = np.zeros((size // CELL, size // CELL), bool)
            for a, b, a2, b2 in zip(x[inside], y[inside], x2[inside], y2[inside]):
                mask[-(-(b - y0) // CELL):(b2 - y0) // CELL, -(-(a - x0) // CELL):(a2 - x0) // CELL] = True  # only cells completely inside the patch
            if mask.mean() >= floor:
                rows.append({"x0": x0, "y0": y0, "size": size, "coverage": float(mask.mean()), "patch_ids": " ".join(sorted(boxes.index[inside]))})
    return pd.DataFrame(rows, columns=["x0", "y0", "size", "coverage", "patch_ids"])


def overlap(a: pd.Series, b: pd.Series) -> float:
    """Intersection over union of two windows of the same size."""
    width, height = max(0, a["size"] - abs(a["x0"] - b["x0"])), max(0, a["size"] - abs(a["y0"] - b["y0"]))
    shared = width * height
    return shared / (2 * a["size"] ** 2 - shared)


def select(candidates: pd.DataFrame, max_overlap: float = 0.5) -> pd.DataFrame:
    """The best-covered window of each set of patches, then, from the best covered down, only windows that do not overlap a chosen one by more than
    ``max_overlap``: near copies of one window would count the same tissue many times."""
    chosen = []
    for _, window in candidates.sort_values(["coverage", "y0", "x0"], ascending=[False, True, True]).drop_duplicates("patch_ids").iterrows():
        if all(overlap(window, other) <= max_overlap for other in chosen):
            chosen.append(window)
    return pd.DataFrame(chosen, columns=candidates.columns).reset_index(drop=True)


def most_severe(labels) -> str:
    """The most severe of the labels: a region with an OSCC patch is OSCC, even if it also has dysplasia."""
    return CLASSES[max(CLASSES.index(label) for label in labels)]


def masked(crop: np.ndarray, boxes: pd.DataFrame, x0: int, y0: int) -> np.ndarray:
    """The crop with everything outside the whole patches filled with the mean colour of the patch pixels, so all the tissue left has a known label."""
    mask = np.zeros(crop.shape[:2], bool)
    for _, box in boxes.iterrows():
        mask[box["y"] - y0: box["y2"] - y0, box["x"] - x0: box["x2"] - x0] = True
    filled = crop.copy()
    filled[~mask] = crop[mask].mean(axis=0).round().astype(np.uint8)
    return filled


def predict_regions(model_name: str, model: torch.nn.Module, preprocessing: dict, regions: pd.DataFrame, device: str, kind: str, baseline: str,
                    pictures: dict, boxes: pd.DataFrame) -> list[dict]:
    """The prediction of the model for each region, unmasked and masked (``kind`` and ``baseline`` are not used: the signature is the one of
    ``src.occlusion.explain``, so it runs through ``explain_on_device``)."""
    transform, _ = transform_and_white(preprocessing)
    model, rows = model.to(device).eval(), []
    for _, region in regions.iterrows():
        x0, y0, size = region["x0"], region["y0"], region["size"]
        crop = pictures[region["histopath_id"]][y0: y0 + size, x0: x0 + size]
        images = {"unmasked": crop, "masked": masked(crop, boxes.loc[region["patch_ids"].split()], x0, y0)}
        batch = torch.stack([transform(Image.fromarray(images[option])) for option in OPTIONS])
        with torch.no_grad():
            probabilities = model(batch.to(device)).softmax(dim=1).cpu().numpy()
        for option, p in zip(OPTIONS, probabilities):
            rows.append({"region_id": region["region_id"], "model": model_name, "option": option, "predicted": CLASSES[int(p.argmax())]} | {f"p_{name}": float(v) for name, v in zip(CLASSES, p)})
    logger.info(f"{model_name}: {len(regions)} regions")
    return rows


def build_regions(panel: pd.DataFrame, boxes: pd.DataFrame, pictures: dict, sizes, step: int, floor: float) -> pd.DataFrame:
    """All the regions of the test pictures, with their label (the most severe whole patch), the labels of their patches and the diagnosis of the picture."""
    rows = []
    for picture, image in pictures.items():
        patches = boxes[boxes["histopath_id"] == picture]
        for size in sizes:
            for n, region in select(windows(patches, image.shape[1], image.shape[0], size, step, floor)).iterrows():
                ids = region["patch_ids"].split()
                labels = panel.loc[ids, "label"]
                rows.append(region.to_dict() | {"region_id": f"{picture}_{size}_{n}", "histopath_id": picture, "n_patches": len(ids), "region_label": most_severe(labels),
                                                "parent": panel.loc[ids[0], "parent"]} | {f"patches_{name}": int((labels == name).sum()) for name in CLASSES})
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--results", type=Path, required=True, help="The folder of Experiment 2 in results/experiments, with panel_patches.csv and representative_checkpoints.csv; the tables go there.")
    parser.add_argument("--experiment", default="ndb_ufes_exp2", help="The MLflow experiment of the cached test predictions (for the predictions of the patches inside a region).")
    parser.add_argument("--models", nargs="+", default=MODELS)
    parser.add_argument("--sizes", nargs="+", type=int, default=list(SIZES))
    parser.add_argument("--step", type=int, default=32)
    parser.add_argument("--floor", type=float, default=0.7, help="The least share of a region that its whole patches have to cover.")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--artifacts", type=Path, default=MLFLOW / "var/mlflow/artifacts")
    args = parser.parse_args()

    panel = pd.read_csv(args.results / "panel_patches.csv", index_col="patch_id", keep_default_na=False)
    lab = pd.read_csv(LAB_TABLE, dtype=str, keep_default_na=False).set_index("patch_id").loc[panel.index]
    boxes = lab[["histopath_id", "x", "y", "x2", "y2"]].astype({column: int for column in ("x", "y", "x2", "y2")})
    paths = lab.groupby("histopath_id")["sab_picture_path"].first()
    pictures = {picture: np.asarray(Image.open(ORGANIZER / path).convert("RGB")) for picture, path in paths.items()}
    regions = build_regions(panel, boxes, pictures, args.sizes, args.step, args.floor)
    logger.info(f"{len(regions)} regions in {regions['histopath_id'].nunique()} of {len(pictures)} pictures; by size: {regions['size'].value_counts().sort_index().to_dict()}")

    checkpoints = pd.read_csv(args.results / "representative_checkpoints.csv").set_index("model")
    cached = prepare(load_predictions(args.experiment))
    device, rows = resolve_device(args.device), []
    for model in args.models:
        own = cached[(cached["model"] == model) & (cached["validation_fold"] == int(checkpoints.loc[model, "fold"]))].set_index("patch_id")["y_pred"]
        regions[f"patches_predicted_max_{model}"] = [CLASSES[max(own.loc[ids.split()])] for ids in regions["patch_ids"]]  # what this model copy says about the patches inside
        path = args.artifacts / checkpoints.loc[model, "checkpoint"]
        if not path.exists():
            raise SystemExit(f"Checkpoint missing, bring it back from S3 with the commands of src.select_checkpoints: {path}")
        rows += explain_on_device(model, path, regions, device, "logit", "white", work=partial(predict_regions, pictures=pictures, boxes=boxes))[0]
    predictions = pd.DataFrame(rows).merge(regions, on="region_id")
    regions.to_csv(args.results / "regions.csv", index=False)
    predictions[["region_id", "model", "option", "predicted", *[f"p_{name}" for name in CLASSES]]].round(4).to_csv(args.results / "region_predictions.csv", index=False)

    predictions["right_region"] = predictions["predicted"] == predictions["region_label"]
    predictions["right_parent"] = predictions["predicted"] == predictions["parent"]
    predictions["same_as_patches_max"] = [row["predicted"] == row[f"patches_predicted_max_{row['model']}"] for _, row in predictions.iterrows()]
    for label, part in (("all regions", predictions), ("regions with coverage >= 0.8", predictions[predictions["coverage"] >= 0.8])):
        print(f"\n{label}: share of regions where the prediction equals the region label, the diagnosis of the picture, and the most severe prediction of the patches inside")
        print(part.groupby(["model", "option", "size"]).agg(regions=("region_id", "nunique"), region_label=("right_region", "mean"), picture=("right_parent", "mean"),
                                                            patches_max=("same_as_patches_max", "mean")).round(3).to_string())
    paired = predictions.pivot_table(index=["region_id", "model"], columns="option", values="predicted", aggfunc="first")
    print("\nmasked and unmasked give the same class:", paired.eq(paired["unmasked"], axis=0)["masked"].groupby("model").mean().round(3).to_dict())
    print("\npredicted class (columns) against the region label (rows), all sizes:")
    print(pd.crosstab([predictions["model"], predictions["option"], predictions["region_label"]], predictions["predicted"]).to_string())


if __name__ == "__main__":
    main()
