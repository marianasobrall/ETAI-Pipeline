"""
Entry point for the baseline predictive pipeline.

Run with:
    python main.py

This orchestrates the full (deliberately simple) pipeline:
    load config -> load data -> preprocess -> split -> train
    -> evaluate (train & test) -> save results
"""

import yaml

from src.data import load_data
from src.preprocessing import clean_and_split, build_preprocessor
from src.model import build_model
from src.evaluate import evaluate, fairness_report
from src.results import save_run
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score

def load_config(path: str = "config.yaml") -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def main():
    config = load_config()
    df = load_data(config["data"]["path"])

    X_train, X_test, y_train, y_test, extras_test = clean_and_split(df, config)

    pipeline = Pipeline([
        ("prep", build_preprocessor(config["preprocessing"])),
        ("model", build_model(config["model"])),
    ])
    pipeline.fit(X_train, y_train)

    y_train_pred = pipeline.predict(X_train)
    y_test_pred = pipeline.predict(X_test)


    cv_scores = cross_val_score(pipeline, X_train, y_train, cv=5, scoring="accuracy")

    cv_summary = (
        f"\nCross-validation (5-fold on training data):\n"
        f"  Scores per fold: {[round(s, 3) for s in cv_scores]}\n"
        f"  Mean accuracy: {cv_scores.mean():.3f}  (std: {cv_scores.std():.3f})\n"
        f"Holdout test accuracy (single split): {pipeline.score(X_test, y_test):.3f}\n"
    )
    

    report = evaluate(y_train, y_train_pred, y_test, y_test_pred)
    report += "\n" + cv_summary
    report += "\n" + fairness_report(
        y_test, y_test_pred, extras_test, sensitive_attr=config["data"]["sensitive_attr"]
    )

    results_dir = config.get("output", {}).get("results_dir", "results")
    path = save_run(results_dir, config, report)
    print(f"Full results saved to {path}")


if __name__ == "__main__":
    main()
