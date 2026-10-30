"""Evaluation: scorers, cross-validation with out-of-fold probabilities, reports, fairness."""
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy import stats
from sklearn.base import clone
from sklearn.metrics import (
    classification_report, confusion_matrix, fbeta_score, get_scorer, make_scorer,
    precision_score, recall_score, f1_score,
)


def build_scorers(metrics: list) -> dict:
    """Metric name -> sklearn scorer. `fbeta_<beta>` weights recall beta times as much as precision."""
    scorers = {}
    for name in metrics:
        if name.startswith("fbeta_"):
            scorers[name] = make_scorer(fbeta_score, beta=float(name.split("_", 1)[1]), zero_division=0)
        elif name == "precision":
            scorers[name] = make_scorer(precision_score, zero_division=0)
        elif name == "recall":
            scorers[name] = make_scorer(recall_score, zero_division=0)
        elif name == "f1":
            scorers[name] = make_scorer(f1_score, zero_division=0)
        else:
            scorers[name] = get_scorer(name)
    return scorers


def score_fold(estimator, X_train, y_train, X_val, y_val, scorers: dict, selection_metric: str):
    """Fits a fresh copy on the training rows; returns this fold's scores and validation probabilities."""
    fitted = clone(estimator).fit(X_train, y_train)
    scores = {name: scorer(fitted, X_val, y_val) for name, scorer in scorers.items()}
    scores["train"] = scorers[selection_metric](fitted, X_train, y_train)
    scores["validation"] = scores[selection_metric]
    scores["gap"] = scores["train"] - scores["validation"]
    return scores, fitted.predict_proba(X_val)[:, 1]


def cross_validate_pipeline(pipeline, X, y, cv, scorers: dict, selection_metric: str, n_jobs: int = 1):
    """
    CV of the whole pipeline. Returns one row per fold (every metric, plus the selection metric's
    train / validation / gap) and the out-of-fold probability for every row of X.
    """
    splits = list(cv.split(X, y))
    results = Parallel(n_jobs=n_jobs)(
        delayed(score_fold)(pipeline, X.iloc[tr], y.iloc[tr], X.iloc[va], y.iloc[va], scorers, selection_metric)
        for tr, va in splits
    )
    oof_proba = np.empty(len(y))
    for (_, va), (_, proba) in zip(splits, results):
        oof_proba[va] = proba
    fold_scores = pd.DataFrame([scores for scores, _ in results],
                               index=pd.RangeIndex(1, len(splits) + 1, name="fold"))
    return fold_scores, oof_proba


def cv_report(fold_scores: pd.DataFrame, metrics: list, selection_metric: str, title: str) -> str:
    """Per-fold table of every metric, then mean ± std, and the selection metric's train-validation gap."""
    table = fold_scores[metrics].copy()
    table.loc["mean"] = fold_scores[metrics].mean()
    table.loc["std"] = fold_scores[metrics].std(ddof=1)
    lines = [
        title,
        "-" * len(title),
        table.to_string(float_format=lambda v: f"{v:.3f}"),
        "",
        f"{selection_metric}: {fold_scores['validation'].mean():.3f} ± {fold_scores['validation'].std(ddof=1):.3f}"
        f"   train: {fold_scores['train'].mean():.3f}   gap (train - validation): {fold_scores['gap'].mean():+.3f}",
    ]
    return "\n".join(lines)


def oof_classification_report(y, oof_proba, threshold: float = 0.5) -> str:
    """Classification report + confusion matrix of the out-of-fold predictions (every row predicted once,
    by a model that never saw it)."""
    y_pred = (oof_proba >= threshold).astype(int)
    cm = pd.DataFrame(confusion_matrix(y, y_pred),
                      index=["true 0", "true 1"], columns=["predicted 0", "predicted 1"])
    return "\n".join([
        f"Out-of-fold predictions (threshold {threshold})",
        classification_report(y, y_pred, zero_division=0),
        "Confusion matrix:",
        cm.to_string(),
        "",
    ])


def fairness_report(y, proba, extras: pd.DataFrame, sensitive_attr: str = "race", threshold: float = 0.5) -> str:
    """
    Deliberately simple fairness check -- not a substitute for a real audit, just enough to show that "accuracy" and "fair" are not the same thing.

    For each group, the false positive rate (did NOT reoffend, predicted to) and false negative rate
    (did reoffend, predicted not to) for:
        - our own model (out-of-fold probabilities at `threshold`)
        - COMPAS's own risk score (score_text != "Low" counts as a "high risk" prediction), for comparison
    """
    df = extras.copy()
    df["y_true"] = np.asarray(y)
    df["y_pred_model"] = (np.asarray(proba) >= threshold).astype(int)
    columns = [("Our model", "y_pred_model")]
    if "score_text" in df.columns:
        df["y_pred_compas"] = (df["score_text"] != "Low").astype(int)
        columns.append(("COMPAS's own score", "y_pred_compas"))

    lines = [
        f"Error rates by {sensitive_attr}",
        "FPR = share of people who did NOT reoffend, but were predicted to",
        "FNR = share of people who did reoffend, but were predicted not to",
        "",
    ]
    for label, col in columns:
        lines.append(f"  {label}:")
        for group, g in df.groupby(sensitive_attr):
            negatives, positives = g[g["y_true"] == 0], g[g["y_true"] == 1]
            fpr = (negatives[col] == 1).mean() if len(negatives) else float("nan")
            fnr = (positives[col] == 0).mean() if len(positives) else float("nan")
            lines.append(f"    {group:<20s} FPR = {fpr:.2f}  FNR = {fnr:.2f}  (n={len(g)})")
        lines.append("")
    return "\n".join(lines)


def corrected_paired_ttest(scores_a, scores_b, n_train: int, n_val: int):
    """
    Nadeau & Bengio's corrected paired t-test for two models scored on the same CV folds.
    The plain t-test is too optimistic because the folds' training sets overlap; the variance
    is inflated by n_val / n_train to account for it. Returns (mean difference, t, two-sided p).
    """
    diff = np.asarray(scores_a) - np.asarray(scores_b)
    k = len(diff)
    var = diff.var(ddof=1)
    if var == 0:
        return diff.mean(), float("nan"), float("nan")
    t = diff.mean() / np.sqrt((1 / k + n_val / n_train) * var)
    p = 2 * stats.t.sf(abs(t), df=k - 1)
    return diff.mean(), t, p
