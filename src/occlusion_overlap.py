import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

from src import logger
from src.config import resolve_device
from src.occlusion import FILLS, UNITS, explain_on_device, output_prefix, patch_sets, pixel_map
from src.panel import patch_table
from src.select_checkpoints import MLFLOW, MODELS

PATCH = 512  # the side of a patch, in pixels of the picture
BLOCK = 32  # the maps are averaged in blocks of this size before they are compared, so that neighbouring pixels do not count as many observations


def positions(meta: pd.DataFrame) -> pd.DataFrame:
    """The patch table with the box of each patch in its picture as numbers."""
    return meta.assign(**{column: meta[column].astype(int) for column in ("x", "y", "x2", "y2")})


def overlap_in_picture(meta: pd.DataFrame, anchor: str) -> pd.Series:
    """For every other patch of the picture of ``anchor``, the share of the area of the anchor that it covers."""
    a = meta.loc[anchor]
    same = meta[(meta["histopath_id"] == a["histopath_id"]) & (meta.index != anchor)]
    width = (np.minimum(same["x2"], a["x2"]) - np.maximum(same["x"], a["x"])).clip(lower=0)
    height = (np.minimum(same["y2"], a["y2"]) - np.maximum(same["y"], a["y"])).clip(lower=0)
    return width * height / PATCH**2


def far_patch(meta: pd.DataFrame, anchor: str, rng: np.random.Generator, max_overlap: float = 0.1) -> str | None:
    """A random patch of the picture of ``anchor`` that has (almost) no tissue in common with it, or None: same picture, same stain, other tissue."""
    overlap = overlap_in_picture(meta, anchor)
    far = overlap.index[overlap < max_overlap]
    return str(rng.choice(far)) if len(far) else None


def neighbour_pairs(meta: pd.DataFrame, anchors: list[str], min_overlap: float, per_anchor: int) -> pd.DataFrame:
    """For each anchor patch, the patches of the same picture that share at least ``min_overlap`` of its area (the ones that share most, ``per_anchor`` at
    most). A pair is listed once."""
    rows, seen = [], set()
    for anchor in anchors:
        overlap = overlap_in_picture(meta, anchor)
        for partner, share in overlap[overlap >= min_overlap].sort_values(ascending=False).head(per_anchor).items():
            if frozenset((anchor, partner)) not in seen:
                seen.add(frozenset((anchor, partner)))
                rows.append({"anchor": anchor, "partner": partner, "overlap": round(float(share), 3)})
    return pd.DataFrame(rows, columns=["anchor", "partner", "overlap"])


def content_correlation(grid_a: np.ndarray, grid_b: np.ndarray, a: pd.Series, b: pd.Series) -> tuple[float, int]:
    """The Spearman correlation between the two maps over the tissue the two patches share, both put in the coordinates of the picture: is the same tissue
    as important in one patch as in the other? Also the number of blocks compared (NaN and 0 when the shared part is smaller than 4 x 4 blocks)."""
    x0, y0, x1, y1 = max(a["x"], b["x"]), max(a["y"], b["y"]), min(a["x2"], b["x2"]), min(a["y2"], b["y2"])
    width, height = (x1 - x0) // BLOCK * BLOCK, (y1 - y0) // BLOCK * BLOCK
    if width < 4 * BLOCK or height < 4 * BLOCK:
        return np.nan, 0
    blocks = []
    for grid, patch in ((grid_a, a), (grid_b, b)):
        crop = pixel_map(grid, PATCH, signed=True)[y0 - patch["y"]: y0 - patch["y"] + height, x0 - patch["x"]: x0 - patch["x"] + width]
        blocks.append(crop.reshape(height // BLOCK, BLOCK, width // BLOCK, BLOCK).mean(axis=(1, 3)).ravel())
    return float(stats.spearmanr(*blocks).statistic), len(blocks[0])


def position_correlation(grid_a: np.ndarray, grid_b: np.ndarray) -> float:
    """The Spearman correlation between the two maps in the coordinates of the patches: is the same place of the patch important in both?"""
    return float(stats.spearmanr(grid_a.ravel(), grid_b.ravel()).statistic)


def pair_table(grids: dict[str, np.ndarray], meta: pd.DataFrame, pairs: pd.DataFrame, seed: int = 0) -> pd.DataFrame:
    """For each pair of neighbouring patches, how well the maps of one model agree when they are aligned by tissue (``content_rho``) and when they are
    aligned by place in the patch (``position_rho``). The baselines of the second: ``same_picture_position_rho`` with a patch of the same picture that
    shares no tissue with the anchor (the ``control`` column; this is what the stain and the kind of tissue of a picture give), and
    ``random_position_rho`` with a random patch of another picture (what a pattern that is the same in every patch gives)."""
    rng, ids = np.random.default_rng(seed), np.array(list(grids))
    rows = []
    for _, pair in pairs.iterrows():
        a, b = meta.loc[pair["anchor"]], meta.loc[pair["partner"]]
        others = ids[meta.loc[ids, "histopath_id"].to_numpy() != a["histopath_id"]]
        content, blocks = content_correlation(grids[pair["anchor"]], grids[pair["partner"]], a, b)
        control = pair.get("control")
        rows.append(pair.to_dict() | {"blocks": blocks, "content_rho": content, "position_rho": position_correlation(grids[pair["anchor"]], grids[pair["partner"]]),
                                      "same_picture_position_rho": position_correlation(grids[pair["anchor"]], grids[control]) if control in grids else np.nan,
                                      "random_position_rho": position_correlation(grids[pair["anchor"]], grids[rng.choice(others)])})
    return pd.DataFrame(rows).round(3)


def overlap_figure(table: pd.DataFrame, path: Path) -> None:
    """Content against position, one point for each pair of patches: high on the vertical axis means the same tissue matters wherever it sits; high on the
    horizontal axis means the same place in the patch matters whatever tissue is there."""
    models = list(dict.fromkeys(table["model"]))
    figure, axes = plt.subplots(1, len(models), figsize=(4.2 * len(models), 4), sharex=True, sharey=True, squeeze=False)
    for axis, model in zip(axes[0], models):
        for group, rows in table[table["model"] == model].groupby("group"):
            axis.scatter(rows["position_rho"], rows["content_rho"], s=14, alpha=0.7, label=group)
        axis.axhline(0, color="grey", lw=0.5)
        axis.axvline(0, color="grey", lw=0.5)
        axis.set_title(model.replace("_patch4_window7_224", ""), fontsize=9)
        axis.set_xlabel("same place in the patch (Spearman)")
    axes[0][0].set_ylabel("same tissue (Spearman)")
    axes[0][-1].legend(fontsize=7)
    figure.tight_layout()
    figure.savefig(path, dpi=130)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--results", type=Path, required=True, help="The folder of an experiment in results/experiments, with panel_patches.csv and representative_checkpoints.csv; the table goes there.")
    parser.add_argument("--models", nargs="+", default=MODELS)
    parser.add_argument("--per-group", type=int, default=30, help="Anchor patches in each group (the same patches as src.occlusion, with the same seed).")
    parser.add_argument("--per-picture", type=int, default=3)
    parser.add_argument("--all-patches", action="store_true", help="Every patch of the test fold is an anchor; the files have _all in their names.")
    parser.add_argument("--min-overlap", type=float, default=0.5, help="The share of its area that a neighbour has to share with the anchor.")
    parser.add_argument("--neighbours", type=int, default=3, help="At most this many neighbours for each anchor.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--score", choices=list(UNITS), default="logit")
    parser.add_argument("--baseline", choices=list(FILLS), default="white")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--artifacts", type=Path, default=MLFLOW / "var/mlflow/artifacts")
    args = parser.parse_args()

    panel = pd.read_csv(args.results / "panel_patches.csv", index_col="patch_id", keep_default_na=False)
    checkpoints = pd.read_csv(args.results / "representative_checkpoints.csv").set_index("model")
    meta = positions(patch_table()).loc[panel.index]  # neighbours and controls only from the test fold: in Experiment 1 the neighbours of a test patch can be training patches
    chosen = patch_sets(panel, args.all_patches, args.per_group, args.per_picture, args.seed)
    group_of = {patch_id: group for group, ids in chosen.items() for patch_id in ids}
    pairs = neighbour_pairs(meta, list(group_of), args.min_overlap, args.neighbours)
    pairs["group"] = pairs["anchor"].map(group_of)
    rng = np.random.default_rng(args.seed)
    pairs["control"] = [far_patch(meta, anchor, rng) for anchor in pairs["anchor"]]
    run_ids = sorted(set(pairs["anchor"]) | set(pairs["partner"]) | set(pairs["control"].dropna()))
    logger.info(f"{len(group_of)} anchor patches, {len(pairs)} pairs, {len(run_ids)} patches to run")
    paths = {model: args.artifacts / checkpoints.loc[model, "checkpoint"] for model in args.models}
    missing = [str(path) for path in paths.values() if not path.exists()]
    if missing:
        raise SystemExit("Checkpoints missing, bring them back from S3 with the commands of src.select_checkpoints:\n" + "\n".join(missing))
    device, tables = resolve_device(args.device), []
    for model, path in paths.items():
        rows, _ = explain_on_device(model, path, {"neighbouring patches": run_ids}, device, args.score, args.baseline)
        tables.append(pair_table({r["patch_id"]: r["grid"] for r in rows}, meta, pairs, args.seed).assign(model=model))
    table = pd.concat(tables, ignore_index=True)
    prefix = output_prefix(args.score, args.baseline, args.all_patches)
    table.to_csv(args.results / f"{prefix}_overlap_pairs.csv", index=False)
    overlap_figure(table, args.results / f"{prefix}_overlap.png")
    print(table.groupby(["model", "group"]).agg(pairs=("anchor", "size"), content_rho=("content_rho", "mean"), position_rho=("position_rho", "mean"),
                                                same_picture_position_rho=("same_picture_position_rho", "mean"), random_position_rho=("random_position_rho", "mean")).round(3).to_string())


if __name__ == "__main__":
    main()
