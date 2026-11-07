import argparse
from functools import partial
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image

from src import logger
from src.config import resolve_device
from src.occlusion import FILLS, GRID, PATCHES, UNITS, explain_on_device, occlude, output_prefix, patch_sets, scores, transform_and_white
from src.select_checkpoints import MLFLOW, MODELS

KS = (1, 3, 5)  # how many windows are covered at once
RANDOM_DRAWS = 10  # random sets of k windows for each patch, the baseline of the test


def cover(image: torch.Tensor, windows, fill: torch.Tensor) -> torch.Tensor:
    """The image with the given windows of the GRID x GRID map (numbered row by row) covered with ``fill``."""
    covered, size = image.clone(), image.shape[-1]
    window, step = size // 4, size // 8
    for k in windows:
        i, j = divmod(int(k), GRID)
        covered[:, i * step: i * step + window, j * step: j * step + window] = fill
    return covered


def deletion(model: torch.nn.Module, image: torch.Tensor, fill: torch.Tensor, device: str, kind: str, rng: np.random.Generator, ks=KS, draws: int = RANDOM_DRAWS,
             delete_fill: torch.Tensor | None = None) -> list[dict]:
    """The deletion test for one patch: its occlusion map ranks the windows; covering the k most important ones should change the prediction more than
    covering k random ones, and much more than covering the k least important. For each k and condition: the share of the covered images whose
    predicted class changed, and the mean drop of the score of the original class. The map is made by covering with ``fill``; the windows are deleted with
    ``delete_fill`` (the same by default). Deleting with another fill keeps the test from repeating the measurement that ranked the windows."""
    delete_fill = fill if delete_fill is None else delete_fill
    full, covered = occlude(model, image, fill, device)
    predicted = int(full.argmax())
    grid = scores(full, kind)[predicted] - scores(covered, kind)[:, :, predicted]
    order = np.argsort(grid.ravel(), kind="stable")[::-1]  # the most important window first
    sets = []
    for k in ks:
        sets += [(k, "top", order[:k]), (k, "bottom", order[-k:])]
        sets += [(k, "random", rng.choice(GRID * GRID, size=k, replace=False)) for _ in range(draws)]
    batch = torch.stack([cover(image, windows, delete_fill) for _, _, windows in sets])
    with torch.no_grad():
        logits = torch.cat([model(chunk.to(device)).cpu() for chunk in batch.split(50)]).numpy()
    changed = logits.argmax(axis=1) != predicted
    drop = scores(full, kind)[predicted] - scores(logits, kind)[:, predicted]
    table = pd.DataFrame({"k": [k for k, _, _ in sets], "condition": [c for _, c, _ in sets], "changed": changed, "drop": drop})
    summary = table.groupby(["k", "condition"]).agg(changed=("changed", "mean"), drop=("drop", "mean")).reset_index()
    return summary.assign(predicted=predicted).to_dict("records")


def deletion_rows(model_name: str, model: torch.nn.Module, preprocessing: dict, chosen: dict[str, list[str]], device: str, kind: str, baseline: str, seed: int = 0,
                  delete_baseline: str | None = None) -> list[dict]:
    """``deletion`` for every patch of every group (same arguments as ``src.occlusion.explain``, so it runs through ``explain_on_device``)."""
    transform, white = transform_and_white(preprocessing)
    model, rng, rows = model.to(device).eval(), np.random.default_rng(seed), []
    for group, patch_ids in chosen.items():
        for patch_id in patch_ids:
            image = transform(Image.open(PATCHES / f"{patch_id}.png").convert("RGB"))
            fills = {"white": white, "mean": image.mean(dim=(1, 2), keepdim=True)}
            rows += [{"model": model_name, "group": group, "patch_id": patch_id} | row
                     for row in deletion(model, image, fills[baseline], device, kind, rng, delete_fill=fills[delete_baseline or baseline])]
        logger.info(f"{model_name}: {group}, {len(patch_ids)} patches")
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--results", type=Path, required=True, help="The folder of an experiment in results/experiments, with panel_patches.csv and representative_checkpoints.csv; the table goes there.")
    parser.add_argument("--models", nargs="+", default=MODELS)
    parser.add_argument("--per-group", type=int, default=30)
    parser.add_argument("--all-patches", action="store_true", help="Every patch of the test fold instead of the four groups (the label, picture diagnosis and n_correct of each patch are added to the table).")
    parser.add_argument("--per-picture", type=int, default=3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--score", choices=list(UNITS), default="logit")
    parser.add_argument("--baseline", choices=list(FILLS), default="white", help="What covers a window when the map is made.")
    parser.add_argument("--delete-baseline", choices=list(FILLS), help="What covers the windows in the deletion (default: the same as --baseline).")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--artifacts", type=Path, default=MLFLOW / "var/mlflow/artifacts")
    args = parser.parse_args()

    panel = pd.read_csv(args.results / "panel_patches.csv", index_col="patch_id", keep_default_na=False)
    checkpoints = pd.read_csv(args.results / "representative_checkpoints.csv").set_index("model")
    chosen = patch_sets(panel, args.all_patches, args.per_group, args.per_picture, args.seed)
    paths = {model: args.artifacts / checkpoints.loc[model, "checkpoint"] for model in args.models}
    missing = [str(path) for path in paths.values() if not path.exists()]
    if missing:
        raise SystemExit("Checkpoints missing, bring them back from S3 with the commands of src.select_checkpoints:\n" + "\n".join(missing))
    device, rows = resolve_device(args.device), []
    for model, path in paths.items():
        rows += explain_on_device(model, path, chosen, device, args.score, args.baseline, work=partial(deletion_rows, seed=args.seed, delete_baseline=args.delete_baseline))[0]
    table = pd.DataFrame(rows).round(4).merge(panel[["label", "parent", "n_correct", "histopath_id"]], left_on="patch_id", right_index=True)
    crossed = args.delete_baseline and args.delete_baseline != args.baseline
    name = f"{output_prefix(args.score, args.baseline, args.all_patches)}_deletion{'_with_' + args.delete_baseline if crossed else ''}.csv"
    table.to_csv(args.results / name, index=False)
    summary = table.groupby(["model", "k", "condition"]).agg(patches=("patch_id", "nunique"), changed=("changed", "mean"), drop=("drop", "mean")).round(3)
    print("Share of covered images whose predicted class changed, and mean drop of the score of the original class:")
    print(summary.unstack("condition").to_string())


if __name__ == "__main__":
    main()
