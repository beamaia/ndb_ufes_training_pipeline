import argparse
from itertools import combinations
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scikit_posthocs as sp
from scipy import stats

from src.validate_input.params import MODEL_FAMILY

METRIC = "test_patch/balanced_acc"
ALPHA = 0.05
SELECT_FROM = ["cnn", "transformer"]  # the families that give one model each to the explainability work


def fold_scores(folds: pd.DataFrame) -> pd.DataFrame:
    """The balanced accuracy of each model (rows) in each fold run (columns), from the ``fold_metrics.csv`` of an experiment."""
    return folds.pivot(index="model", columns="fold", values=METRIC)


def family_of(model: str) -> str:
    return MODEL_FAMILY.get(model, "other")


def assumptions(scores: pd.DataFrame) -> dict[str, float]:
    """The checks of the article before choosing the test: normality of each model's scores (Shapiro-Wilk) and equal spread across models (Levene)."""
    not_normal = sum(stats.shapiro(row).pvalue < ALPHA for row in scores.to_numpy())
    return {"not_normal": int(not_normal), "levene_p": float(stats.levene(*scores.to_numpy()).pvalue)}


def kruskal_conover(scores: pd.DataFrame) -> tuple[float, float, pd.DataFrame]:
    """The method of the article: Kruskal-Wallis over the models (every fold score is a separate observation), then Conover-Iman for every pair with the
    Holm-Sidak correction. Returns the statistic, the p-value and the matrix of adjusted p-values between models.

    It treats the five scores of a model as independent, which they are not quite: the fold models share most of their training data. The p-values lean
    small for that reason, so ``friedman`` and ``nemenyi`` below are kept as the paired check.
    """
    statistic, p_value = stats.kruskal(*scores.to_numpy())
    long = scores.stack().rename("score").reset_index()
    return float(statistic), float(p_value), sp.posthoc_conover(long, val_col="score", group_col="model", p_adjust="holm-sidak")


def families_compared(scores: pd.DataFrame) -> dict[str, float]:
    """CNN against transformer, all the fold scores of each family together (the 'architecture comparison' of the article)."""
    pooled = {family: scores[[family_of(model) == family for model in scores.index]].to_numpy().ravel() for family in SELECT_FROM}
    statistic, p_value = stats.mannwhitneyu(pooled["cnn"], pooled["transformer"], alternative="two-sided")
    return {"cnn_median": float(np.median(pooled["cnn"])), "transformer_median": float(np.median(pooled["transformer"])), "u": float(statistic), "p": float(p_value),
            "cnn_n": len(pooled["cnn"]), "transformer_n": len(pooled["transformer"])}


def friedman(scores: pd.DataFrame) -> tuple[float, float]:
    """The paired check: the Friedman test on the folds (every fold is a block, every model a treatment)."""
    statistic, p_value = stats.friedmanchisquare(*scores.to_numpy())
    return float(statistic), float(p_value)


def nemenyi(scores: pd.DataFrame) -> tuple[pd.Series, pd.DataFrame, float]:
    """The paired follow-up to Friedman: the mean rank of each model (in each fold, rank 1 is the best), the p-value of every pair, and the critical
    difference, the distance between mean ranks that two models need to be told apart. Wilcoxon cannot do this with five folds: its smallest p is 0.0625."""
    k, n = scores.shape
    mean_rank = scores.rank(ascending=False).mean(axis=1)
    scale = np.sqrt(k * (k + 1) / (12 * n))
    pairs = pd.DataFrame([{"model_1": a, "model_2": b, "rank_gap": abs(mean_rank[a] - mean_rank[b])} for a, b in combinations(scores.index, 2)])
    pairs["p_nemenyi"] = stats.studentized_range.sf(pairs["rank_gap"] / scale, k, np.inf)
    return mean_rank, pairs, float(stats.studentized_range.ppf(1 - ALPHA, k, np.inf) * scale)


def ranking_table(scores: pd.DataFrame, mean_rank: pd.Series) -> pd.DataFrame:
    table = pd.DataFrame({"family": [family_of(model) for model in scores.index], "mean_rank": mean_rank, "balanced_acc_mean": scores.mean(axis=1) * 100,
                          "balanced_acc_std": scores.std(axis=1, ddof=0) * 100})
    return table.sort_values("mean_rank").round(3)


def pair_table(pairs: pd.DataFrame, conover: pd.DataFrame) -> pd.DataFrame:
    pairs = pairs.assign(p_conover_holm_sidak=[conover.loc[a, b] for a, b in zip(pairs["model_1"], pairs["model_2"])])
    return pairs.round(4)


def p_between(pairs: pd.DataFrame, a: str, b: str, column: str = "p_conover_holm_sidak") -> float:
    match = pairs[((pairs["model_1"] == a) & (pairs["model_2"] == b)) | ((pairs["model_1"] == b) & (pairs["model_2"] == a))]
    return float(match[column].iloc[0])


def verdict(p_value: float) -> str:
    return "the models differ" if p_value < ALPHA else f"no difference between the models was detected at alpha={ALPHA}"


def selection(ranking: pd.DataFrame, pairs: pd.DataFrame, checks: dict, folds: int) -> str:
    """The text that goes with the tables: the tests, and the best model of each family with the models it cannot be told apart from (by the article's method)."""
    kw, fr, fam = checks["kruskal"], checks["friedman"], checks["families"]
    lines = [f"{len(ranking)} models, {folds} fold runs each, balanced accuracy on the test fold.",
             f"Checks: {checks['assumptions']['not_normal']} of {len(ranking)} models have scores that are not normal (Shapiro-Wilk); equal spread across models, Levene p={checks['assumptions']['levene_p']:.3f}.",
             f"Kruskal-Wallis (method of the article): statistic {kw[0]:.3f}, p={kw[1]:.4f}; {verdict(kw[1])}. Pairs: Conover-Iman with Holm-Sidak.",
             f"Friedman (paired check): statistic {fr[0]:.3f}, p={fr[1]:.4f}; {verdict(fr[1])}. Nemenyi critical difference {checks['critical_difference']:.2f} rank units (ranks go from 1 to {len(ranking)}).",
             f"CNN against transformer, all fold scores of each family together ({fam['cnn_n']} and {fam['transformer_n']} scores): median {fam['cnn_median'] * 100:.1f}% against {fam['transformer_median'] * 100:.1f}%, Mann-Whitney p={fam['p']:.4f}.", ""]
    for family in SELECT_FROM:
        group = ranking[ranking["family"] == family]
        if group.empty:
            continue
        best = group["balanced_acc_mean"].idxmax()
        tied = [model for model in group.index if model != best and p_between(pairs, best, model) >= ALPHA]
        lines.append(f"- best {family}: {best} ({group.loc[best, 'balanced_acc_mean']:.1f}%); cannot be told apart from: {', '.join(tied) if tied else 'none'}")
    overall = ranking["balanced_acc_mean"].idxmax()
    beaten = [model for model in ranking.index if model != overall and p_between(pairs, overall, model) < ALPHA]
    beaten_paired = [model for model in ranking.index if model != overall and p_between(pairs, overall, model, "p_nemenyi") < ALPHA]
    lines.append(f"- best overall: {overall}; better than (Conover): {', '.join(beaten) if beaten else 'none'}; better than (Nemenyi, paired): {', '.join(beaten_paired) if beaten_paired else 'none'}")
    return "\n".join(lines) + "\n"


def critical_difference_plot(mean_rank: pd.Series, conover: pd.DataFrame, path: Path) -> None:
    """The critical-difference diagram of the article: models on a rank line, joined by a bar when they cannot be told apart."""
    figure, axis = plt.subplots(figsize=(10, 4))
    sp.critical_difference_diagram(mean_rank, conover, ax=axis)
    figure.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--results", type=Path, required=True, help="The folder of an experiment in results/experiments, with fold_metrics.csv; the tables are written there.")
    args = parser.parse_args()
    scores = fold_scores(pd.read_csv(args.results / "fold_metrics.csv"))
    statistic, p_value, conover = kruskal_conover(scores)
    mean_rank, pairs, critical_difference = nemenyi(scores)
    ranking, pair_rows = ranking_table(scores, mean_rank), pair_table(pairs, conover)
    ranking.to_csv(args.results / "model_ranking.csv", index_label="model")
    pair_rows.to_csv(args.results / "model_pairs.csv", index=False)
    critical_difference_plot(mean_rank, conover, args.results / "model_critical_difference.png")
    checks = {"assumptions": assumptions(scores), "kruskal": (statistic, p_value), "friedman": friedman(scores), "families": families_compared(scores), "critical_difference": critical_difference}
    text = selection(ranking, pair_rows, checks, scores.shape[1])
    (args.results / "model_selection.md").write_text(text)
    print(text)


if __name__ == "__main__":
    main()
