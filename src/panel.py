import argparse
import os
import shutil
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from scipy import stats
from sklearn.metrics import roc_auc_score, roc_curve

from src import logger
from src.config import load_env

REPO = Path(__file__).resolve().parents[1]
ORGANIZER = REPO.parent / "ndb_ufes_data_organizer"
CACHE = REPO / "notebooks" / "cache"
CLASSES = ["Without dysplasia", "With dysplasia", "OSCC"]  # the class numbers 0, 1, 2, from the least to the most severe
PROBABILITIES = ["prob_leukoplakia_without_dysplasia", "prob_leukoplakia_with_dysplasia", "prob_oscc"]
LABEL = {"Leukoplakia without dysplasia": "Without dysplasia", "Leukoplakia with dysplasia": "With dysplasia", "OSCC": "OSCC"}
NOT_IN_NDB = "not in NDB-UFES"
PARENTS = {  # for each patch label, the diagnoses of the parent picture that the label-authority table shows
    "With dysplasia": ["OSCC", "With dysplasia", "Without dysplasia", NOT_IN_NDB],
    "OSCC": ["OSCC", "With dysplasia", NOT_IN_NDB],
    "Without dysplasia": ["With dysplasia", "Without dysplasia", "OSCC", NOT_IN_NDB],
}


def finished_fold_runs(experiment: str) -> pd.DataFrame:
    runs = mlflow.search_runs(experiment_names=[experiment], filter_string="attributes.status = 'FINISHED'")
    return runs[runs["tags.mlflow.parentRunId"].notna()]


def download_predictions(folds: pd.DataFrame, kind: str) -> pd.DataFrame:
    frames = []
    for _, run in folds.iterrows():
        path = mlflow.artifacts.download_artifacts(run_id=run["run_id"], artifact_path=kind, dst_path=str(CACHE / "download" / run["run_id"]))
        frames.append(pd.read_csv(path, keep_default_na=False).assign(run_id=run["run_id"], validation_fold=int(run["tags.validation_fold"])))
    shutil.rmtree(CACHE / "download", ignore_errors=True)
    return pd.concat(frames, ignore_index=True)


def load_predictions(experiment: str, kind: str = "heldout_predictions/predictions.csv") -> pd.DataFrame:
    """The predictions of all the finished fold runs of an experiment on the test fold, from MLflow, kept in notebooks/cache. The cache is used only when
    it holds exactly the finished fold runs MLflow has now (a cache written while an experiment was still training is downloaded again); when MLflow
    cannot be reached, the cache is used as it is, with a warning."""
    cache = CACHE / f"{experiment}.csv"
    try:
        folds = finished_fold_runs(experiment)
    except Exception as error:  # MLflow is not running
        if not cache.exists():
            raise
        logger.warning(f"MLflow cannot be reached ({str(error)[:80]}); using the cache of {experiment} without checking it")
        return pd.read_csv(cache, keep_default_na=False)
    if cache.exists():
        cached = pd.read_csv(cache, keep_default_na=False)
        if "run_id" in cached and set(cached["run_id"]) == set(folds["run_id"]):
            return cached
        logger.info(f"The cache of {experiment} does not hold its {len(folds)} finished fold runs; downloading them again")
    CACHE.mkdir(parents=True, exist_ok=True)
    frame = download_predictions(folds, kind)
    frame.to_csv(cache, index=False)
    return frame


def prepare(frame: pd.DataFrame) -> pd.DataFrame:
    frame["y_true"], frame["y_pred"] = frame["y_true"].astype(int), frame["y_pred"].astype(int)
    return frame


def patch_table(organizer: Path = ORGANIZER) -> pd.DataFrame:
    """One row for each patch: its picture, its patient group, ``label`` (the label of the patch) and ``parent`` (the NDB-UFES diagnosis of its picture)."""
    meta = pd.read_csv(organizer / "results/phase0/dataset_tables/patch_table_public.csv", dtype=str, keep_default_na=False).set_index("patch_id")
    meta["label"] = meta["patch_diagnosis"].map(LABEL)
    meta["parent"] = meta["ndb_diagnosis"].map(LABEL).fillna(NOT_IN_NDB)
    return meta


def panel_of(frame: pd.DataFrame, meta: pd.DataFrame) -> pd.DataFrame:
    """One row for each patch: what each model says (the class with the highest mean probability over its fold runs), and how the models agree.

    ``n_correct`` is how many models get the patch right, ``consensus`` the class most of them say (a tie goes to the less severe class), ``mean_severity`` the
    mean of their class numbers, ``share_oscc`` the share that say OSCC and ``disagreement`` the entropy of their answers in bits. Works on the test
    predictions and on the validation predictions.
    """
    by_model = frame.groupby(["patch_id", "model"])[PROBABILITIES].mean()
    calls = pd.Series(by_model.to_numpy().argmax(axis=1), index=by_model.index).unstack("model")  # patches x models: the class number each model says
    truth = frame.drop_duplicates("patch_id").set_index("patch_id")["y_true"].loc[calls.index]
    shares = np.stack([calls.eq(c).sum(axis=1) / calls.shape[1] for c in range(3)], axis=1)
    panel = pd.DataFrame(index=calls.index)
    panel["true"] = truth.map(dict(enumerate(CLASSES)))
    panel["n_correct"] = calls.eq(truth, axis=0).sum(axis=1)
    panel["consensus"] = pd.Series(shares.argmax(axis=1), index=calls.index).map(dict(enumerate(CLASSES)))
    panel["mean_severity"] = calls.mean(axis=1)
    panel["share_oscc"] = shares[:, 2]
    panel["disagreement"] = stats.entropy(shares.T, base=2)
    panel = panel.join(meta[["label", "parent", "histopath_id", "case_group"]])
    return panel.join(calls.apply(lambda column: column.map(dict(enumerate(CLASSES)))).add_prefix("says_"))


def n_correct_counts(panel: pd.DataFrame) -> pd.DataFrame:
    """For each true class, how many patches are got right by 0, 1, ... models: a mass at both ends means the difficulty is in the patches."""
    n_models = sum(column.startswith("says_") for column in panel.columns)
    return pd.crosstab(panel["true"], panel["n_correct"]).reindex(index=CLASSES, columns=range(n_models + 1), fill_value=0)


def confusion_by_zoom(frame: pd.DataFrame, meta: pd.DataFrame) -> pd.DataFrame:
    """The confusion cells of each model on the test fold, split by the possible zoom of the picture of the patch (10x or 40x, column possible_zoom of
    the patch table), added up over the fold runs as in confusion_matrices.csv, with the share of the true class. ``all models`` adds up every model.
    ``patches`` and ``pictures`` count the test patches and pictures of the true class at that zoom: the patches of one picture are not independent, so a
    share carried by few pictures is weak evidence."""
    names = dict(enumerate(CLASSES))
    cells = frame.assign(zoom=frame["patch_id"].map(meta["possible_zoom"]), true=frame["y_true"].map(names), predicted=frame["y_pred"].map(names))
    if cells["zoom"].isin(["", np.nan]).any():
        raise ValueError(f"{cells.loc[cells['zoom'].isin(['', np.nan]), 'patch_id'].nunique()} test patches have no possible zoom in the patch table")
    both = pd.concat([cells, cells.assign(model="all models")])
    index = pd.MultiIndex.from_product([sorted(cells["model"].unique()) + ["all models"], sorted(cells["zoom"].unique()), CLASSES, CLASSES],
                                       names=["model", "zoom", "true", "predicted"])
    table = both.groupby(["model", "zoom", "true", "predicted"]).size().reindex(index, fill_value=0).rename("count").reset_index()
    table["share of true class %"] = (table["count"] / table.groupby(["model", "zoom", "true"])["count"].transform("sum") * 100).round(1)
    units = cells.drop_duplicates("patch_id").groupby(["zoom", "true"]).agg(patches=("patch_id", "size"), pictures=("histopath_id", "nunique"))
    return table.join(units, on=["zoom", "true"])


def parent_pull_by_zoom(frame: pd.DataFrame, meta: pd.DataFrame) -> pd.DataFrame:
    """For the patches of pictures with an NDB-UFES diagnosis, by patch label, parent diagnosis and possible zoom: what share of the model calls (all
    models, added up over their fold runs) say the patch label, the parent diagnosis, or each class. ``relation`` says whether the label is the same as,
    less severe or more severe than the parent. ``parent class on matched patches %`` is how often that same class is said on patches of the same label
    and zoom whose picture has that label too, and ``excess %`` the difference: how much more the parent's class is said than the zoom and the label
    alone would give."""
    names, severity = dict(enumerate(CLASSES)), {name: k for k, name in enumerate(CLASSES)}
    calls = frame.assign(zoom=frame["patch_id"].map(meta["possible_zoom"]), label=frame["patch_id"].map(meta["label"]), parent=frame["patch_id"].map(meta["parent"]),
                         said=frame["y_pred"].map(names), picture=frame["patch_id"].map(meta["histopath_id"]))
    calls = calls[calls["parent"].isin(CLASSES)]
    rows = []
    for (zoom, label, parent), part in calls.groupby(["zoom", "label", "parent"]):
        shares = {f"said {name} %": (part["said"] == name).mean() * 100 for name in CLASSES}
        rows.append({"zoom": zoom, "patch label": label, "parent diagnosis": parent,
                     "relation": "same" if label == parent else "less severe than parent" if severity[label] < severity[parent] else "more severe than parent",
                     "patches": part["patch_id"].nunique(), "pictures": part["picture"].nunique(), "said label %": shares[f"said {label} %"],
                     "said parent %": shares[f"said {parent} %"], **shares})
    table = pd.DataFrame(rows)
    matched = table[table["relation"] == "same"].set_index(["zoom", "patch label"])
    table["parent class on matched patches %"] = [matched[f"said {r['parent diagnosis']} %"].get((r["zoom"], r["patch label"]), np.nan) for _, r in table.iterrows()]
    table["excess %"] = table["said parent %"] - table["parent class on matched patches %"]
    return table.round(1)


def parent_vs_label(meta: pd.DataFrame) -> pd.DataFrame:
    """How the label of a patch relates to the diagnosis of its picture: patches, and the share of each parent diagnosis and of each patch label."""
    counts = pd.crosstab(meta["parent"], meta["label"]).reindex(index=[*CLASSES, NOT_IN_NDB], columns=CLASSES, fill_value=0)
    rows = counts.stack().rename("patches").reset_index()
    rows["% of the parent diagnosis"] = rows["patches"] / rows.groupby("parent")["patches"].transform("sum") * 100
    rows["% of the patch label"] = rows["patches"] / rows.groupby("label")["patches"].transform("sum") * 100
    return rows.round(1)


def label_authority(panel: pd.DataFrame) -> pd.DataFrame:
    """The patches of each label, by the diagnosis of their parent picture: what the models say about them. If the models call OSCC much more often when
    the parent is OSCC, they follow the parent against the label of the patch. ``by picture`` takes the mean per picture first, which shows how much a few
    pictures carry the result."""
    rows = []
    for label, parents in PARENTS.items():
        for parent in parents:
            group = panel[(panel["label"] == label) & (panel["parent"] == parent)]
            rows.append({"patch label": label, "parent diagnosis": parent, "patches": len(group), "pictures": group["histopath_id"].nunique(),
                         "models saying OSCC, %": group["share_oscc"].mean() * 100, "consensus OSCC, % of patches": (group["consensus"] == "OSCC").mean() * 100,
                         "mean severity": group["mean_severity"].mean(), "mean severity, by picture": group.groupby("histopath_id")["mean_severity"].mean().mean()})
    return pd.DataFrame(rows).round(2)


FPR_GRID = np.linspace(0, 1, 101)


def roc_curves(frame: pd.DataFrame) -> pd.DataFrame:
    """For each model and class (that class against the other two), the ROC curve averaged over the fold runs at the same false-positive rates, and the
    mean and standard deviation of the AUC of the runs (as in summary.csv). ``all classes`` is the mean of the three curves, whose AUC is the macro AUC."""
    rows = []
    for model, runs in frame.groupby("model"):
        curves = {name: [] for name in CLASSES}
        aucs = {name: [] for name in CLASSES}
        for _, run in runs.groupby("validation_fold"):
            for number, name in enumerate(CLASSES):
                positive, score = run["y_true"] == number, run[PROBABILITIES[number]]
                fpr, tpr, _ = roc_curve(positive, score)
                curves[name].append(np.interp(FPR_GRID, fpr, tpr))
                aucs[name].append(roc_auc_score(positive, score))
        curves["all classes"] = [np.mean([curves[name][i] for name in CLASSES], axis=0) for i in range(len(curves[CLASSES[0]]))]
        aucs["all classes"] = list(np.mean([aucs[name] for name in CLASSES], axis=0))
        for name, runs_tpr in curves.items():
            rows.append({"model": model, "class": name, "auc_mean": float(np.mean(aucs[name])), "auc_std": float(np.std(aucs[name])), "tpr": np.mean(runs_tpr, axis=0)})
    return pd.DataFrame(rows)


def roc_table(curves: pd.DataFrame) -> pd.DataFrame:
    """The curves of ``roc_curves`` as a long table (one row for each model, class and false-positive rate), for the slides and other plots."""
    rows = [{"model": r["model"], "class": r["class"], "auc_mean": r["auc_mean"], "auc_std": r["auc_std"], "fpr": fpr, "tpr": tpr}
            for _, r in curves.iterrows() for fpr, tpr in zip(FPR_GRID, r["tpr"])]
    return pd.DataFrame(rows).round(4)


def roc_figure(curves: pd.DataFrame, title: str, path: Path) -> None:
    """An interactive page: a menu picks one class or all classes together, and clicking a model in the legend hides it."""
    views = [*CLASSES, "all classes"]
    figure = go.Figure()
    for view in views:
        for _, row in curves[curves["class"] == view].sort_values("auc_mean", ascending=False).iterrows():
            figure.add_trace(go.Scatter(x=FPR_GRID, y=row["tpr"], mode="lines", visible=view == views[0], legendgroup=row["model"],
                                        name=f"{row['model']} (AUC {row['auc_mean'] * 100:.1f} ± {row['auc_std'] * 100:.1f})"))
    figure.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", line={"dash": "dash", "color": "grey"}, name="chance (AUC 50)", visible=True))
    per_view = len(figure.data) - 1
    shown = lambda view: [curves.loc[curves["class"] == v].shape[0] * [v == view] for v in views]
    buttons = [{"label": view, "method": "update", "args": [{"visible": [x for block in shown(view) for x in block] + [True]},
                                                         {"title": f"{title}: {view} against the rest" if view != "all classes" else f"{title}: mean of the three classes (macro)"}]}
               for view in views]
    figure.update_layout(title=f"{title}: {views[0]} against the rest", xaxis_title="false-positive rate (1 - specificity)", yaxis_title="true-positive rate (sensitivity)",
                         updatemenus=[{"buttons": buttons, "x": 1.0, "y": 1.15, "xanchor": "right"}], autosize=True, legend={"font": {"size": 10}})
    figure.update_xaxes(range=[0, 1])
    figure.update_yaxes(range=[0, 1], scaleanchor="x")
    assert per_view == len(views) * curves["model"].nunique()
    figure.write_html(path, include_plotlyjs=True, default_width="100%", default_height="100%", config={"responsive": True})  # fills its window or a slide


def export(experiment: str, out: Path) -> None:
    meta = patch_table()
    out.mkdir(parents=True, exist_ok=True)
    predictions = prepare(load_predictions(experiment))
    curves = roc_curves(predictions)
    roc_figure(curves, f"ROC curves on the test fold ({experiment}; mean of the fold runs)", out / "roc_curves.html")
    roc_table(curves).to_csv(out / "roc_curves.csv", index=False)
    panel = panel_of(predictions, meta)
    panel.to_csv(out / "panel_patches.csv")
    n_correct_counts(panel).to_csv(out / "panel_n_correct.csv")
    parent_vs_label(meta).to_csv(out / "parent_vs_label.csv", index=False)
    tables, by_zoom = {"test fold": label_authority(panel)}, {"test fold": confusion_by_zoom(predictions, meta)}
    pull = {"test fold": parent_pull_by_zoom(predictions, meta)}
    validation = CACHE / f"{experiment}_validation.csv"  # made by src/validation_predictions.py; covers the five folds that are not the test fold
    if validation.exists():  # each patch predicted by the runs that held its fold out of training: the pattern on the other pictures
        held_out = prepare(pd.read_csv(validation, keep_default_na=False))
        tables["validation folds"] = label_authority(panel_of(held_out, meta))
        by_zoom["validation folds"] = confusion_by_zoom(held_out, meta)
        pull["validation folds"] = parent_pull_by_zoom(held_out, meta)
        by_zoom |= {f"validation fold {fold}": confusion_by_zoom(part, meta) for fold, part in held_out.groupby("validation_fold")}
    by_zoom = pd.concat(by_zoom, names=["predictions on"]).reset_index("predictions on")
    by_zoom.to_csv(out / "confusion_by_zoom.csv", index=False)
    pull = pd.concat(pull, names=["predictions on"]).reset_index("predictions on")
    pull.to_csv(out / "parent_pull_by_zoom.csv", index=False)
    columns = ["patches", "pictures", "said label %", "said parent %", "parent class on matched patches %", "excess %"]
    print(f"{experiment}, patches with a label other than their picture, all models (% of the model calls):")
    print(pull[pull["relation"] != "same"].set_index(["predictions on", "relation", "patch label", "parent diagnosis", "zoom"])[columns].sort_index().to_string())
    for on in ("test fold", "validation folds"):
        pooled = by_zoom[(by_zoom["model"] == "all models") & (by_zoom["predictions on"] == on)]
        if len(pooled):
            print(f"{experiment}, {on}, all models, share of the true class % (rows: zoom, true class, its patches and pictures; columns: predicted):")
            print(pooled.pivot_table(index=["zoom", "true", "patches", "pictures"], columns="predicted", values="share of true class %", sort=False)[CLASSES].to_string())
    pd.concat(tables, names=["predictions on"]).reset_index("predictions on").to_csv(out / "label_authority.csv", index=False)
    print(f"{experiment}: {len(panel)} test patches, {sum(c.startswith('says_') for c in panel.columns)} models -> {out}" + ("" if validation.exists() else " (no validation predictions yet)"))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--experiment", default="ndb_ufes_exp2", help="The MLflow experiment, for example ndb_ufes_exp2.")
    parser.add_argument("--out", type=Path, required=True, help="The folder for the tables, for example ../ndb_ufes_data_organizer/results/experiments/experiment_2.")
    args = parser.parse_args()
    os.environ.setdefault("MLFLOW_ENABLE_ARTIFACTS_PROGRESS_BAR", "false")
    load_env(REPO / ".env", Path(os.getenv("MLFLOW_ENV_FILE", REPO.parent / "ndb_ufes_mlflow" / ".env")))
    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://localhost:8000/"))
    export(args.experiment, args.out)


if __name__ == "__main__":
    main()
