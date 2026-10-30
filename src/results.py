"""
Saving each run's results to disk.

Printing to the terminal is fine while you're watching it happen, but it's gone the moment you scroll past it or close the window. This module writes the full report to `results/run_<id>.txt`, and adds one row per run to a scoreboard CSV, so runs can be compared later after changing something in config.yaml.
"""
import os

import pandas as pd
import yaml


def save_run(results_dir: str, run: str, config: dict, report_text: str) -> str:
    """
    Writes one run's full report (with the config that produced it) to `results_dir/run_<run>.txt`
    (the folder is created if needed) and returns the path that was written.
    """
    os.makedirs(results_dir, exist_ok=True)
    path = os.path.join(results_dir, f"run_{run}.txt")

    header = (
        f"Run: {run}\n"
        f"Model: {config['model']['type']}  params={config['model'].get('params')}\n"
        f"Test set: size {config['test_set']['size']}  random_state {config['test_set']['random_state']}"
        "  (locked, not used)\n"
        + "=" * 60 + "\n\n"
    )
    footer = "\n" + "=" * 60 + "\nConfig used:\n" + yaml.safe_dump(config, sort_keys=False, allow_unicode=True)

    with open(path, "w", encoding="utf-8") as f:
        f.write(header + report_text + footer)

    return path


def save_scores(path: str, row: dict) -> pd.DataFrame:
    """Appends one run's row to the scoreboard CSV (created if missing) and returns every run in it."""
    folder = os.path.dirname(path)
    if folder:
        os.makedirs(folder, exist_ok=True)
    new = pd.DataFrame([row])
    runs = pd.concat([pd.read_csv(path), new], ignore_index=True) if os.path.exists(path) else new
    runs.to_csv(path, index=False)
    return runs
