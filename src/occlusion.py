import argparse
from itertools import combinations
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from PIL import Image
from scipy import special, stats
from torchvision import transforms

from src import logger
from src.checkpoint import load_checkpoint
from src.config import resolve_device
from src.panel import CLASSES, ORGANIZER
from src.select_checkpoints import MLFLOW, MODELS

PATCHES = ORGANIZER / "data/ndb_ufes/patch_level/images"
UNITS = {"probability": "probability", "logit": "log-odds"}
FILLS = {"white": "white", "mean": "the mean colour of the patch"}
GRID = 7  # windows per side: a window is a quarter of the patch and the step an eighth, which is 128 and 64 pixels on a patch of 512


def output_prefix(score: str, baseline: str, all_patches: bool = False) -> str:
    """The start of the names of the files of a run: occlusion, and what is not the default (occlusion_logit_mean_all)."""
    return "_".join(["occlusion"] + (["logit"] if score == "logit" else []) + (["mean"] if baseline == "mean" else []) + (["all"] if all_patches else []))


ALL = "all test patches"


def patch_sets(panel: pd.DataFrame, all_patches: bool, per_group: int, per_picture: int, seed: int) -> dict[str, list[str]]:
    """Every patch of the test fold as one set, or the four groups of ``groups``."""
    return {ALL: list(panel.index)} if all_patches else groups(panel, per_group, per_picture, seed)


def groups(panel: pd.DataFrame, per_group: int, per_picture: int, seed: int) -> dict[str, list[str]]:
    """The four groups of patches the maps are compared on, from the panel of the twelve models, with at most ``per_picture`` patches of one picture
    (the patches of a picture are not independent observations, and one picture should not be the whole group)."""
    n_models = sum(column.startswith("says_") for column in panel.columns)
    with_dysplasia = panel["label"] == "With dysplasia"
    definitions = {
        "agrees with the parent": with_dysplasia & (panel["parent"] == "OSCC") & (panel["consensus"] == "OSCC"),
        "genuine error": with_dysplasia & (panel["parent"] == "With dysplasia") & (panel["consensus"] == "OSCC"),
        "all models right": with_dysplasia & (panel["n_correct"] == n_models),
        "no model right": panel["n_correct"] == 0,
    }
    rng = np.random.default_rng(seed)
    chosen = {}
    for name, mask in definitions.items():
        taken, per_picture_count = [], {}
        for patch_id in rng.permutation(panel.index[mask].to_numpy()):
            picture = panel.loc[patch_id, "histopath_id"]
            if per_picture_count.get(picture, 0) < per_picture:
                taken.append(patch_id)
                per_picture_count[picture] = per_picture_count.get(picture, 0) + 1
            if len(taken) == per_group:
                break
        chosen[name] = taken
    return chosen


def occlude(model: torch.nn.Module, image: torch.Tensor, white: torch.Tensor, device: str, batch_size: int = 50) -> tuple[np.ndarray, np.ndarray]:
    """The class logits of the image, and of the image with each of the GRID x GRID windows covered with white. ``image`` is already normalized and
    ``white`` is the value of white in the same units (one number for each colour channel)."""
    size = image.shape[-1]
    window, step = size // 4, size // 8
    batch = image.unsqueeze(0).repeat(1 + GRID * GRID, 1, 1, 1)
    for k in range(GRID * GRID):
        i, j = divmod(k, GRID)
        batch[1 + k, :, i * step: i * step + window, j * step: j * step + window] = white
    with torch.no_grad():
        logits = torch.cat([model(chunk.to(device)).cpu() for chunk in batch.split(batch_size)])
    return logits[0].numpy(), logits[1:].reshape(GRID, GRID, -1).numpy()


def scores(logits: np.ndarray, kind: str) -> np.ndarray:
    """The score of every class: its probability, or its log-odds (its logit against all the others). A probability near 1 hardly moves when evidence is
    covered, so the log-odds shows where the evidence is for a model that is sure."""
    if kind == "probability":
        return special.softmax(logits, axis=-1)
    others = np.stack([special.logsumexp(np.delete(logits, c, axis=-1), axis=-1) for c in range(logits.shape[-1])], axis=-1)
    return logits - others


def describe(grid: np.ndarray) -> dict[str, float]:
    """Where the evidence is: how much the largest window matters, how focal it is (the share of the total drop in the three largest windows; a uniform
    map gives 3/49), and how much of it is in the middle of the patch (the central 3 x 3 windows; a uniform map gives 9/49)."""
    positive = np.clip(grid, 0, None)
    total = positive.sum()
    top3 = np.sort(positive.ravel())[::-1][:3].sum()
    return {"max_drop": float(grid.max()), "windows_over_0.1": int((grid > 0.1).sum()),
            "focality_top3": float(top3 / total) if total > 0 else np.nan, "centre_share": float(positive[2:5, 2:5].sum() / total) if total > 0 else np.nan}


def pixel_map(grid: np.ndarray, size: int, signed: bool = False) -> np.ndarray:
    """The map at the size of the image: each pixel is the mean drop of the windows that cover it (negative drops count as 0 unless ``signed``)."""
    window, step = size // 4, size // 8
    total, count = np.zeros((size, size)), np.zeros((size, size))
    for i in range(GRID):
        for j in range(GRID):
            total[i * step: i * step + window, j * step: j * step + window] += grid[i, j] if signed else max(grid[i, j], 0)
            count[i * step: i * step + window, j * step: j * step + window] += 1
    return total / count


def transform_and_white(preprocessing: dict) -> tuple[transforms.Compose, torch.Tensor]:
    """The transform the model was trained with, and the value of white after it (one number for each colour channel)."""
    size = preprocessing["size"]
    transform = transforms.Compose([transforms.Resize((size, size)), transforms.ToTensor(), transforms.Normalize(preprocessing["mean"], preprocessing["std"])])
    return transform, ((1 - torch.tensor(preprocessing["mean"])) / torch.tensor(preprocessing["std"])).reshape(3, 1, 1)


def explain(model_name: str, model: torch.nn.Module, preprocessing: dict, chosen: dict[str, list[str]], device: str, kind: str, baseline: str) -> list[dict]:
    """For every patch of every group: the class the model predicts, its probabilities, and the drop of the score of that class (``kind``) for each window
    covered with white, or with the mean colour of the patch (``baseline``)."""
    transform, white = transform_and_white(preprocessing)
    model = model.to(device).eval()
    rows = []
    for group, patch_ids in chosen.items():
        for n, patch_id in enumerate(patch_ids, 1):
            image = transform(Image.open(PATCHES / f"{patch_id}.png").convert("RGB"))
            full, covered = occlude(model, image, white if baseline == "white" else image.mean(dim=(1, 2), keepdim=True), device)
            predicted = int(full.argmax())
            rows.append({"model": model_name, "group": group, "patch_id": patch_id, "predicted": predicted, "probabilities": special.softmax(full),
                         "grid": scores(full, kind)[predicted] - scores(covered, kind)[:, :, predicted]})
        logger.info(f"{model_name}: {group}, {len(patch_ids)} patches")
    return rows


def explain_on_device(model_name: str, path: Path, chosen: dict[str, list[str]], device: str, kind: str, baseline: str, work=None) -> tuple[list[dict], dict]:
    """``explain`` (or ``work``, a function with the same arguments) on the device, and on the cpu if the model has an operation the device (a Mac GPU,
    for example) does not support. Also returns the preprocessing saved in the checkpoint."""
    work = work or explain
    model, bundle = load_checkpoint(path, "cpu", torch.float32)
    for where in ("cpu",) if device == "cpu" else (device, "cpu"):
        try:
            return work(model_name, model, bundle["preprocessing"], chosen, where, kind, baseline), bundle["preprocessing"]
        except RuntimeError as error:
            if where == "cpu":
                raise
            logger.warning(f"{model_name} could not run on {where} ({str(error)[:80]}...); it goes on the cpu")


def summary_table(results: list[dict], panel: pd.DataFrame) -> pd.DataFrame:
    rows = [{"model": r["model"], "group": r["group"], "patch_id": r["patch_id"], "label": panel.loc[r["patch_id"], "label"], "parent": panel.loc[r["patch_id"], "parent"],
             "predicted": CLASSES[r["predicted"]], "p_predicted": float(r["probabilities"][r["predicted"]])} | describe(r["grid"]) for r in results]
    return pd.DataFrame(rows).round(4)


def grid_table(results: list[dict]) -> pd.DataFrame:
    """One row for each window of each patch and model: the drop of the probability when that window is covered (row and column count from the top left)."""
    rows = [{"model": r["model"], "patch_id": r["patch_id"], "row": i, "column": j, "drop": float(r["grid"][i, j])} for r in results for i in range(GRID) for j in range(GRID)]
    return pd.DataFrame(rows).round(4)


def agreement_table(results: list[dict]) -> pd.DataFrame:
    """For every patch and pair of models, the Spearman correlation between their maps: do the models look at the same places?"""
    by_patch = {}
    for r in results:
        by_patch.setdefault((r["group"], r["patch_id"]), {})[r["model"]] = r["grid"].ravel()
    rows = [{"group": group, "patch_id": patch_id, "model_1": a, "model_2": b, "spearman": float(stats.spearmanr(grids[a], grids[b]).statistic)}
            for (group, patch_id), grids in by_patch.items() for a, b in combinations(sorted(grids), 2)]
    return pd.DataFrame(rows).round(3)


def examples_figure(model_name: str, results: list[dict], preprocessing: dict, panel: pd.DataFrame, path: Path, kind: str, baseline: str, columns: int = 5) -> None:
    """The first patches of each group with their map on top, for one model."""
    chosen = [r for r in results if r["model"] == model_name]
    names = list(dict.fromkeys(r["group"] for r in chosen))
    figure, axes = plt.subplots(len(names), columns, figsize=(2.4 * columns, 2.9 * len(names)), squeeze=False)
    for axis in axes.ravel():
        axis.axis("off")
    size = preprocessing["size"]
    for row, name in enumerate(names):
        for column, r in enumerate([r for r in chosen if r["group"] == name][:columns]):
            axis = axes[row][column]
            axis.imshow(Image.open(PATCHES / f"{r['patch_id']}.png").convert("RGB").resize((size, size)))
            axis.imshow(pixel_map(r["grid"], size), cmap="jet", alpha=0.45, vmin=0)
            axis.set_title(f"{r['patch_id']}\nlabel {panel.loc[r['patch_id'], 'label']}; says {CLASSES[r['predicted']]}", fontsize=7)
        axes[row][0].text(-0.05, 1.2, name, transform=axes[row][0].transAxes, fontsize=9, fontweight="bold")
    figure.suptitle(f"Occlusion of {model_name}: drop of the {UNITS[kind]} of the predicted class when a window is covered with {FILLS[baseline]}", fontsize=10)
    figure.tight_layout(rect=(0, 0, 1, 0.97))
    figure.savefig(path, dpi=130)
    plt.close(figure)


def mean_grids_figure(results: list[dict], path: Path) -> None:
    """The mean map of each group for each model: where the evidence falls on average."""
    models, names = list(dict.fromkeys(r["model"] for r in results)), list(dict.fromkeys(r["group"] for r in results))
    means = {(m, g): np.mean([r["grid"] for r in results if r["model"] == m and r["group"] == g], axis=0) for m in models for g in names}
    top = max(np.clip(grid, 0, None).max() for grid in means.values())
    figure, axes = plt.subplots(len(models), len(names), figsize=(2.6 * len(names), 2.6 * len(models)), squeeze=False)
    for row, model_name in enumerate(models):
        for column, name in enumerate(names):
            image = axes[row][column].imshow(np.clip(means[(model_name, name)], 0, None), cmap="jet", vmin=0, vmax=top)
            axes[row][column].set_xticks([])
            axes[row][column].set_yticks([])
            if row == 0:
                axes[row][column].set_title(name, fontsize=8)
        axes[row][0].set_ylabel(model_name.replace("_patch4_window7_224", ""), fontsize=8)
    figure.colorbar(image, ax=axes, shrink=0.8, label="mean drop")
    figure.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--results", type=Path, required=True, help="The folder of an experiment in results/experiments, with panel_patches.csv and representative_checkpoints.csv; the tables and figures go there.")
    parser.add_argument("--models", nargs="+", default=MODELS)
    parser.add_argument("--per-group", type=int, default=30, help="Patches in each of the four groups.")
    parser.add_argument("--all-patches", action="store_true", help="Every patch of the test fold instead of the four groups; the files have _all in their names.")
    parser.add_argument("--per-picture", type=int, default=3, help="At most this many patches of one picture in a group.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--score", choices=list(UNITS), default="probability", help="What the drop is measured in.")
    parser.add_argument("--baseline", choices=list(FILLS), default="white", help="What covers the window. The names of the files say what is not the default: occlusion_logit_mean_*.")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--artifacts", type=Path, default=MLFLOW / "var/mlflow/artifacts", help="Where MLflow keeps the checkpoints (restored from S3).")
    args = parser.parse_args()

    panel = pd.read_csv(args.results / "panel_patches.csv", index_col="patch_id", keep_default_na=False)
    checkpoints = pd.read_csv(args.results / "representative_checkpoints.csv").set_index("model")
    chosen = patch_sets(panel, args.all_patches, args.per_group, args.per_picture, args.seed)
    logger.info("Patches in each group: " + ", ".join(f"{name} {len(ids)}" for name, ids in chosen.items()))
    paths = {model: args.artifacts / checkpoints.loc[model, "checkpoint"] for model in args.models}
    missing = [str(path) for path in paths.values() if not path.exists()]
    if missing:
        raise SystemExit("Checkpoints missing, bring them back from S3 with the commands of src.select_checkpoints:\n" + "\n".join(missing))
    device, results = resolve_device(args.device), []
    prefix = output_prefix(args.score, args.baseline, args.all_patches)
    for model, path in paths.items():
        rows, preprocessing = explain_on_device(model, path, chosen, device, args.score, args.baseline)
        results += rows
        examples_figure(model, results, preprocessing, panel, args.results / f"{prefix}_{model}.png", args.score, args.baseline)
    summary = summary_table(results, panel)
    summary.to_csv(args.results / f"{prefix}_summary.csv", index=False)
    grid_table(results).to_csv(args.results / f"{prefix}_grids.csv", index=False)
    agreement = agreement_table(results)
    agreement.to_csv(args.results / f"{prefix}_agreement.csv", index=False)
    mean_grids_figure(results, args.results / f"{prefix}_mean_maps.png")
    columns = ["p_predicted", "max_drop", "windows_over_0.1", "focality_top3", "centre_share"]
    print(summary.groupby(["model", "group"])[columns].mean().round(3).to_string())
    print("\nSpearman between the maps of two models, mean for each group:")
    print(agreement.groupby(["group", "model_1", "model_2"])["spearman"].mean().round(3).to_string())


if __name__ == "__main__":
    main()
