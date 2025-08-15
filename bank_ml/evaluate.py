"""Evaluation utilities."""
from __future__ import annotations
from pathlib import Path
import json

import pandas as pd
from sklearn.metrics import accuracy_score, classification_report


def evaluate_and_save(model, X_test: pd.DataFrame, y_test: pd.Series, output_dir: Path) -> dict:
    y_pred = model.predict(X_test)
    metrics = {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "report": classification_report(y_test, y_pred, output_dict=True),
    }
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "metrics.json").open("w") as fh:
        json.dump(metrics, fh, indent=2)
    return metrics
