"""Evaluation utilities for trained models."""

from __future__ import annotations

from pathlib import Path
from typing import Dict

import json

import numpy as np
import pandas as pd
from loguru import logger

try:  # pragma: no cover - matplotlib optional
    from matplotlib import pyplot as plt  # type: ignore
except Exception:  # pragma: no cover
    plt = None
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    mean_squared_error,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
    ConfusionMatrixDisplay,
)

from .config import Config


def _save_confusion_matrix(cm: np.ndarray, name: str) -> None:
    """Save a confusion matrix plot under the ``assets`` directory."""

    if plt is None:  # pragma: no cover - plotting optional
        return
    assets = Path("assets")
    assets.mkdir(exist_ok=True)
    try:  # pragma: no cover - plotting not essential for tests
        disp = ConfusionMatrixDisplay(confusion_matrix=cm)
        disp.plot()
        plt.title(f"{name} Confusion Matrix")
        plt.tight_layout()
        plt.savefig(assets / f"{name}_confusion_matrix.png")
        plt.close()
    except Exception as exc:  # pragma: no cover - ignore plotting errors
        logger.warning("Failed to plot confusion matrix for %s: %s", name, exc)


def _save_curves(y_test: np.ndarray, y_proba: np.ndarray, name: str) -> None:
    """Save ROC and PR curves for binary classifiers."""

    if y_proba.shape[1] != 2:
        return

    if plt is None:  # pragma: no cover - plotting optional
        return
    assets = Path("assets")
    assets.mkdir(exist_ok=True)

    try:  # ROC curve
        fpr, tpr, _ = roc_curve(y_test, y_proba[:, 1])
        plt.figure()
        plt.plot(fpr, tpr, label="ROC")
        plt.plot([0, 1], [0, 1], linestyle="--", color="grey")
        plt.xlabel("FPR")
        plt.ylabel("TPR")
        plt.title(f"{name} ROC curve")
        plt.tight_layout()
        plt.savefig(assets / f"{name}_roc_curve.png")
        plt.close()

        prec, rec, _ = precision_recall_curve(y_test, y_proba[:, 1])
        plt.figure()
        plt.plot(rec, prec, label="PR")
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.title(f"{name} PR curve")
        plt.tight_layout()
        plt.savefig(assets / f"{name}_pr_curve.png")
        plt.close()
    except Exception as exc:  # pragma: no cover - ignore plotting issues
        logger.warning("Failed to plot ROC/PR curves for %s: %s", name, exc)


def evaluate_models(
    models: Dict[str, object],
    X_test: np.ndarray,
    y_test: np.ndarray,
    cfg: Config,
) -> pd.DataFrame:
    """Evaluate a dictionary of models on a test set.

    The function computes a range of classification metrics, creates basic
    diagnostic plots and stores a summary table in ``cfg.paths.output_dir`` as
    both CSV and Markdown.
    """

    rows = []
    for name, model in models.items():
        y_pred = model.predict(X_test)

        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average="weighted", zero_division=0)
        recall = recall_score(y_test, y_pred, average="weighted", zero_division=0)
        f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)
        mse = mean_squared_error(y_test, y_pred.astype(int))

        roc_auc = np.nan
        if hasattr(model, "predict_proba"):
            try:
                y_proba = model.predict_proba(X_test)
                roc_auc = roc_auc_score(
                    y_test, y_proba, multi_class="ovr", average="macro"
                )
                _save_curves(y_test, y_proba, name)
            except Exception:
                roc_auc = np.nan

        cm = confusion_matrix(y_test, y_pred)
        _save_confusion_matrix(cm, name)

        rows.append(
            {
                "model": name,
                "accuracy": accuracy,
                "precision_weighted": precision,
                "recall_weighted": recall,
                "f1_weighted": f1,
                "roc_auc_ovr_macro": roc_auc,
                "mean_squared_error": mse,
            }
        )

    df = pd.DataFrame(rows).set_index("model")

    out_dir = Path(cfg.paths.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_dir / "metrics.csv")
    with (out_dir / "metrics.md").open("w") as fh:
        try:  # pandas.to_markdown requires the optional tabulate dep
            fh.write(df.to_markdown())
        except Exception:  # pragma: no cover - fallback when tabulate missing
            fh.write(df.to_string())

    return df


__all__ = ["evaluate_models"]

