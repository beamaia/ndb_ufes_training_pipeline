"""Do the models look at hematoxylin-dense regions (the dark purple) more than their size predicts?

For each test patch and model: the occlusion importance of each of the 7 x 7 windows (the positive part of the drop of the log-odds of the predicted class, from
``src.occlusion --score logit --all-patches``) is weighed by the share of that window that is hematoxylin-dense. The preference is the importance share inside
the dense regions divided by the share of the windows they cover: 1 means no preference, above 1 that the evidence is on the dense regions more than their size
predicts. Both shares are taken over the same windows, so the uneven overlap of the windows does not matter.

The dense regions are the pixels whose hematoxylin concentration (colour deconvolution of Ruifrok and Johnston) is above ONE threshold for the whole dataset: the
Otsu threshold of the pooled values of every patch, never one per patch (a per-patch Otsu adapts to the patch and its fractions would not be comparable). They
are called hematoxylin-dense regions; nothing here says that they are nuclei.

Written to ``--results`` (the folder of an experiment): ``hematoxylin_preference.csv`` (one row for each model and patch), ``hematoxylin_summary.csv`` and
``hematoxylin_preference.png``.

With ``--intervals`` the tables above are read, not made again: the patches of one picture are not independent, so the median preference of each group gets a 95%
interval from resampling the pictures (each with all its patches), in ``hematoxylin_intervals.csv``. Given two folders, the difference of their medians gets an
interval too (the first minus the second, independent resamples), in ``hematoxylin_difference.csv`` of the first folder.
"""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image

from src import logger
from src.occlusion import GRID, PATCHES
from src.panel import patch_table

STAINS = np.array([[0.65, 0.70, 0.29], [0.07, 0.99, 0.11], [0.27, 0.57, 0.78]])  # hematoxylin, eosin and the rest, as optical density vectors (Ruifrok and Johnston 2001)
UNMIX = np.linalg.inv(STAINS / np.linalg.norm(STAINS, axis=1, keepdims=True))
SIZE = 128  # the patches are looked at in this size (a quarter of 512), so a window is 32 pixels and the step 16
BINS = 512  # the histogram of the pooled hematoxylin values, from which the threshold is taken


def hematoxylin(rgb: np.ndarray) -> np.ndarray:
    """The hematoxylin concentration of each pixel of an RGB image (uint8), never below 0."""
    density = -np.log10((rgb.astype(np.float32) + 1) / 256)
    return np.clip(density @ UNMIX[:, 0], 0, None)


def load(path: Path) -> np.ndarray:
    """The hematoxylin map of a patch, reduced to SIZE x SIZE by averaging."""
    with Image.open(path) as image:
        return hematoxylin(np.asarray(image.convert("RGB").resize((SIZE, SIZE), Image.Resampling.BOX)))


def otsu(values: np.ndarray, bins: int = BINS) -> float:
    """The Otsu threshold of a set of values: the cut that gives the largest variance between the two groups it makes."""
    counts, edges = np.histogram(values, bins=bins)
    centres = (edges[:-1] + edges[1:]) / 2
    weight_low, weight_high = np.cumsum(counts), counts.sum() - np.cumsum(counts)
    mean_low = np.cumsum(counts * centres) / np.maximum(weight_low, 1)
    mean_high = (np.sum(counts * centres) - np.cumsum(counts * centres)) / np.maximum(weight_high, 1)
    between = weight_low * weight_high * (mean_low - mean_high) ** 2
    best = np.flatnonzero(between >= between.max() * (1 - 1e-9))  # when the two groups are apart the variance is the same for every cut in the gap: take the middle of it
    return float(edges[int(best.mean()) + 1])


def window_fractions(mask: np.ndarray) -> np.ndarray:
    """The share of each of the GRID x GRID windows (a quarter of the patch, in steps of an eighth, as in src.occlusion) that is inside the mask."""
    size = mask.shape[0]
    window, step = size // 4, size // 8
    return np.array([[mask[i * step: i * step + window, j * step: j * step + window].mean() for j in range(GRID)] for i in range(GRID)])


def preference(drops: np.ndarray, fractions: np.ndarray) -> dict[str, float]:
    """The share of the importance inside the dense regions, the share of the windows they cover, and their ratio. NaN when no window matters or no window is dense."""
    importance = np.clip(drops, 0, None)
    inside = float((importance * fractions).sum() / importance.sum()) if importance.sum() > 0 else np.nan
    area = float(fractions.mean())
    return {"importance_share": inside, "area_share": area, "preference": inside / area if area > 0 else np.nan}


def preference_table(grids: pd.DataFrame, masks: dict[str, np.ndarray]) -> pd.DataFrame:
    """One row for each model and patch of the grid table (columns model, patch_id, row, column, drop)."""
    rows = []
    for (model, patch_id), group in grids.groupby(["model", "patch_id"]):
        drops = np.zeros((GRID, GRID))
        drops[group["row"].to_numpy(), group["column"].to_numpy()] = group["drop"].to_numpy()
        rows.append({"model": model, "patch_id": patch_id} | preference(drops, window_fractions(masks[patch_id])))
    return pd.DataFrame(rows)


def describe(table: pd.DataFrame, panel: pd.DataFrame, meta: pd.DataFrame) -> pd.DataFrame:
    """The preference table with the label of each patch, whether the model got it right, and its possible zoom."""
    table = table.join(meta[["possible_zoom"]], on="patch_id").rename(columns={"possible_zoom": "zoom"})
    table = table.join(panel[["true"]], on="patch_id").rename(columns={"true": "label"})
    said = np.array([panel.at[patch, f"says_{model}"] for patch, model in zip(table["patch_id"], table["model"])])
    table["right"] = said == table["label"].to_numpy()
    return table


def summary(table: pd.DataFrame, threshold: float) -> pd.DataFrame:
    """For each model (and all together) and each way to group the patches: the patches, the median preference, the share of patches above 1, and the mean shares."""
    rows = []
    for kind, column in [("all", None), ("right", "right"), ("label", "label"), ("zoom", "zoom")]:
        for model, part in [("all models", table)] + list(table.groupby("model")):
            groups = [("all", part)] if column is None else list(part.groupby(column))
            for value, group in groups:
                rows.append({"model": model, "grouped by": kind, "group": value, "patches": len(group), "median preference": group["preference"].median(),
                             "share above 1": (group["preference"] > 1).mean(), "mean importance share": group["importance_share"].mean(),
                             "mean area share": group["area_share"].mean(), "threshold": threshold})
    return pd.DataFrame(rows)


def figure(table: pd.DataFrame, path: Path) -> None:
    """The preference of every patch, by model, for right and wrong calls, by label and by zoom; the dashed line is no preference."""
    models = sorted(table["model"].unique())
    panels = [("right", "the model's call", {True: "right", False: "wrong"}), ("label", "label of the patch", {}), ("zoom", "possible zoom", {})]
    figure, axes = plt.subplots(1, len(panels), figsize=(15, 4.5), sharey=True)
    for axis, (column, title, names) in zip(axes, panels):
        groups = sorted(table[column].unique(), key=str)
        width = 0.8 / len(models)
        for k, model in enumerate(models):
            data = [table[(table["model"] == model) & (table[column] == g)]["preference"].dropna() for g in groups]
            positions = np.arange(len(groups)) + (k - (len(models) - 1) / 2) * width
            boxes = axis.boxplot(data, positions=positions, widths=width * 0.9, patch_artist=True, showfliers=False)
            for box in boxes["boxes"]:
                box.set_facecolor(f"C{k}")
        axis.axhline(1, color="grey", linestyle="--")
        axis.set_xticks(np.arange(len(groups)), [names.get(g, g) for g in groups])
        axis.set_title(title)
    axes[0].set_ylabel("importance share on dense regions / area share (1 = no preference)")
    axes[0].legend([plt.Rectangle((0, 0), 1, 1, color=f"C{k}") for k in range(len(models))], models, fontsize=8)
    figure.tight_layout()
    figure.savefig(path, dpi=130)
    plt.close(figure)


def picture_bootstrap(values: pd.Series, pictures: pd.Series, draws: int, seed: int) -> np.ndarray:
    """The median of ``values`` in ``draws`` resamples of the pictures: each draw picks as many pictures as there are, with replacement, and takes all their values."""
    groups = [group.dropna().to_numpy() for _, group in values.groupby(pictures)]
    groups = [g for g in groups if len(g)]
    rng = np.random.default_rng(seed)
    return np.array([np.median(np.concatenate([groups[k] for k in rng.integers(len(groups), size=len(groups))])) for _ in range(draws)])


def group_draws(table: pd.DataFrame, draws: int, seed: int) -> dict[tuple, dict]:
    """For each model (and all together) and way of grouping, as in ``summary``: the median, the resampled medians, the patches and the pictures. ``table`` has ``picture``."""
    result = {}
    for kind, column in [("all", None), ("right", "right"), ("label", "label"), ("zoom", "zoom")]:
        for model, part in [("all models", table)] + list(table.groupby("model")):
            for value, group in ([("all", part)] if column is None else list(part.groupby(column))):
                result[(model, kind, value)] = {"median": group["preference"].median(), "draws": picture_bootstrap(group["preference"], group["picture"], draws, seed),
                                                "patches": len(group), "pictures": group["picture"].nunique()}
    return result


def intervals(table: pd.DataFrame, draws: int = 2000, seed: int = 0) -> pd.DataFrame:
    """The median preference of each group with the 2.5th and 97.5th percentiles of its resampled medians."""
    rows = []
    for (model, kind, value), found in group_draws(table, draws, seed).items():
        low, high = np.percentile(found["draws"], [2.5, 97.5])
        rows.append({"model": model, "grouped by": kind, "group": value, "patches": found["patches"], "pictures": found["pictures"], "median preference": found["median"], "low": low, "high": high})
    return pd.DataFrame(rows)


def difference(first: pd.DataFrame, second: pd.DataFrame, draws: int = 2000, seed: int = 0) -> pd.DataFrame:
    """The median preference of ``first`` minus that of ``second`` for every group in both, with the percentiles of the differences of independent resamples."""
    one, two = group_draws(first, draws, seed), group_draws(second, draws, seed + 1)
    rows = []
    for key in one.keys() & two.keys():
        gap = one[key]["draws"] - two[key]["draws"]
        low, high = np.percentile(gap, [2.5, 97.5])
        rows.append({"model": key[0], "grouped by": key[1], "group": key[2], "pictures first": one[key]["pictures"], "pictures second": two[key]["pictures"],
                     "difference": one[key]["median"] - two[key]["median"], "low": low, "high": high})
    return pd.DataFrame(rows).sort_values(["grouped by", "model", "group"], key=lambda s: s.astype(str)).reset_index(drop=True)


def with_pictures(folder: Path, meta: pd.DataFrame) -> pd.DataFrame:
    """The preference table of an experiment with the picture of each patch."""
    return pd.read_csv(folder / "hematoxylin_preference.csv", keep_default_na=False, na_values=[""]).join(meta[["histopath_id"]].rename(columns={"histopath_id": "picture"}), on="patch_id")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--results", type=Path, nargs="+", required=True, help="The folder of an experiment in results/experiments, with panel_patches.csv and the occlusion grids; the tables go there. With --intervals, one or two folders with hematoxylin_preference.csv.")
    parser.add_argument("--grids", default="occlusion_logit_all_grids.csv", help="The occlusion grids of the test patches: the log-odds drop, covered with white (src.occlusion --score logit --all-patches).")
    parser.add_argument("--intervals", action="store_true", help="Intervals that resample the pictures, from the tables already made (and, with two folders, the difference between them).")
    parser.add_argument("--draws", type=int, default=2000, help="Resamples for the intervals.")
    args = parser.parse_args()
    meta = patch_table()
    if args.intervals:
        tables = [with_pictures(folder, meta) for folder in args.results]
        for folder, table in zip(args.results, tables):
            result = intervals(table, args.draws)
            result.round(3).to_csv(folder / "hematoxylin_intervals.csv", index=False)
            print(f"{folder.name}: median preference with a 95% interval from resampling the pictures\n" + result[result["model"] == "all models"].round(3).to_string(index=False))
        if len(tables) == 2:
            gap = difference(tables[0], tables[1], args.draws)
            gap.round(3).to_csv(args.results[0] / "hematoxylin_difference.csv", index=False)
            print(f"{args.results[0].name} minus {args.results[1].name}:\n" + gap[gap["model"] == "all models"].round(3).to_string(index=False))
        return
    if len(args.results) != 1:
        parser.error("give one folder, or two with --intervals")
    folder = args.results[0]
    logger.info(f"Hematoxylin maps of the {len(meta)} patches of the dataset, to set one threshold for all of them")
    maps = {patch_id: load(PATCHES / f"{patch_id}.png") for patch_id in meta.index}
    threshold = otsu(np.concatenate([m.ravel() for m in maps.values()]))
    covered = float(np.mean([(m > threshold).mean() for m in maps.values()]))
    logger.info(f"Threshold: hematoxylin concentration above {threshold:.3f} (Otsu over the pooled pixels of every patch); the dense regions cover {covered:.1%} of a patch on average")
    panel = pd.read_csv(folder / "panel_patches.csv", index_col="patch_id", keep_default_na=False)
    grids = pd.read_csv(folder / args.grids)
    table = describe(preference_table(grids, {p: maps[p] > threshold for p in grids["patch_id"].unique()}), panel, meta)
    table.round(4).to_csv(folder / "hematoxylin_preference.csv", index=False)
    overview = summary(table, threshold)
    overview.round(3).to_csv(folder / "hematoxylin_summary.csv", index=False)
    figure(table, folder / "hematoxylin_preference.png")
    print(overview[overview["model"] == "all models"].round(3).to_string(index=False))


if __name__ == "__main__":
    main()
