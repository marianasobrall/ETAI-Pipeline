"""Hyperparameter tuning with Optuna, and nested CV to estimate the tuned model honestly."""
import numpy as np
import optuna
import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import cross_val_score

from src.evaluate import score_fold


def _suggest(trial, name: str, spec: dict):
    kind = spec["type"]
    if kind == "int":
        return trial.suggest_int(name, spec["low"], spec["high"], log=spec.get("log", False))
    if kind == "float":
        return trial.suggest_float(name, spec["low"], spec["high"], log=spec.get("log", False))
    if kind == "categorical":
        return trial.suggest_categorical(name, spec["choices"])
    raise ValueError(f"Unknown search space type for {name}: {kind}. Options: int, float, categorical")


def tune_pipeline(pipeline, X, y, inner_cv, scorer, search_space: dict, n_trials: int,
                  random_state: int, n_jobs: int = 1):
    """
    Optuna search (TPE sampler): each candidate is scored by `inner_cv` on (X, y).
    Returns an unfitted copy of the pipeline with the best hyperparameters set, and the study.
    """
    def objective(trial):
        params = {name: _suggest(trial, name, spec) for name, spec in search_space.items()}
        candidate = clone(pipeline).set_params(**params)
        return cross_val_score(candidate, X, y, cv=inner_cv, scoring=scorer, n_jobs=n_jobs).mean()

    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=random_state))
    study.optimize(objective, n_trials=n_trials)
    return clone(pipeline).set_params(**study.best_params), study


def nested_cross_validate(pipeline, X, y, outer_cv, inner_cv, scorers: dict, selection_metric: str,
                          search_space: dict, n_trials: int, random_state: int, n_jobs: int = 1):
    """
    Nested CV: in every outer fold the whole tuning procedure runs on the outer-training rows only,
    and the model it picks is scored on the outer-validation rows it never saw. Same return shape
    as cross_validate_pipeline; each fold's chosen hyperparameters are kept in fold_scores.attrs.
    """
    rows, fold_params, inner_best = [], [], []
    oof_proba = np.empty(len(y))
    for fold, (tr, va) in enumerate(outer_cv.split(X, y), start=1):
        X_tr, y_tr = X.iloc[tr], y.iloc[tr]
        tuned, study = tune_pipeline(pipeline, X_tr, y_tr, inner_cv, scorers[selection_metric],
                                     search_space, n_trials, random_state, n_jobs=n_jobs)
        scores, proba = score_fold(tuned, X_tr, y_tr, X.iloc[va], y.iloc[va], scorers, selection_metric)
        rows.append(scores)
        oof_proba[va] = proba
        fold_params.append(study.best_params)
        inner_best.append(study.best_value)
        print(f"  outer fold {fold}: {selection_metric} = {scores['validation']:.3f}", flush=True)

    fold_scores = pd.DataFrame(rows, index=pd.RangeIndex(1, len(rows) + 1, name="fold"))
    fold_scores.attrs["best_params"] = fold_params
    fold_scores.attrs["inner_best"] = inner_best
    return fold_scores, oof_proba


def _format_params(params: dict) -> str:
    return ", ".join(f"{name.replace('model__', '')}={value:.4g}" if isinstance(value, float)
                     else f"{name.replace('model__', '')}={value}" for name, value in params.items())


def tuning_report(study, nested_scores: pd.DataFrame, selection_metric: str, top: int = 5) -> str:
    """Chosen hyperparameters, how optimistic the inner score is, each outer fold's pick, and the best trials."""
    nested_mean = nested_scores["validation"].mean()
    lines = [
        "Tuning (Optuna, TPE sampler)",
        "----------------------------",
        f"Chosen on all development rows: {_format_params(study.best_params)}",
        f"Best inner-CV {selection_metric}: {study.best_value:.3f}   nested estimate: {nested_mean:.3f}   "
        f"optimism of picking the best trial: {study.best_value - nested_mean:+.3f}",
    ]
    if "best_params" in nested_scores.attrs:
        lines += ["", "Picked in each outer fold (inner best -> outer validation):"]
        for fold, (params, inner, outer) in enumerate(
                zip(nested_scores.attrs["best_params"], nested_scores.attrs["inner_best"],
                    nested_scores["validation"]), start=1):
            lines.append(f"  fold {fold}: {inner:.3f} -> {outer:.3f}   {_format_params(params)}")

    trials = sorted((t for t in study.trials if t.value is not None), key=lambda t: t.value, reverse=True)
    lines += ["", f"Top {min(top, len(trials))} of {len(study.trials)} trials (all development rows):"]
    for t in trials[:top]:
        lines.append(f"  #{t.number:<3d} {t.value:.3f}   {_format_params(t.params)}")
    return "\n".join(lines)
