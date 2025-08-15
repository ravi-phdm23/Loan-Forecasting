"""Model building and training utilities."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Tuple

import joblib
import numpy as np
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.base import clone
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils.class_weight import compute_class_weight
from sklearn.pipeline import Pipeline as SkPipeline
from sklearn.preprocessing import StandardScaler

from .config import Config
from .optimize_pso import pso_tune_mlp


# ---------------------------------------------------------------------------
# Simple pipeline used by the CLI
# ---------------------------------------------------------------------------


def build_model(imbalance: str) -> SkPipeline:
    class_weight = "balanced" if imbalance == "class_weight" else None
    clf = LogisticRegression(max_iter=200, class_weight=class_weight)
    pipe = SkPipeline([("scaler", StandardScaler()), ("model", clf)])
    return pipe


# ---------------------------------------------------------------------------
# Helper construction utilities
# ---------------------------------------------------------------------------


def make_mlp(params: Dict[str, object], cfg: Config) -> MLPClassifier:
    """Create an :class:`MLPClassifier` from PSO parameters and config."""

    return MLPClassifier(
        **params,
        solver="sgd",
        early_stopping=True,
        max_iter=300,
        random_state=cfg.cv.random_state,
    )


def _handle_imbalance(X: np.ndarray, y: np.ndarray, cfg: Config) -> Tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    """Resample or compute sample weights depending on imbalance strategy."""

    sample_weight = None
    if cfg.imbalance == "smote":
        sampler = SMOTE(random_state=cfg.cv.random_state)
        X, y = sampler.fit_resample(X, y)
    elif cfg.imbalance == "class_weight":
        classes = np.unique(y)
        weights = compute_class_weight("balanced", classes=classes, y=y)
        mapping = dict(zip(classes, weights))
        sample_weight = np.array([mapping[c] for c in y])
    return X, y, sample_weight


# ---------------------------------------------------------------------------
# Baseline factories
# ---------------------------------------------------------------------------


def _make_baselines(cfg: Config) -> Dict[str, object]:
    weight = "balanced" if cfg.imbalance == "class_weight" else None
    rs = cfg.cv.random_state
    return {
        "log_reg": LogisticRegression(max_iter=200, class_weight=weight),
        "decision_tree": DecisionTreeClassifier(class_weight=weight, random_state=rs),
        "random_forest": RandomForestClassifier(
            n_estimators=200,
            class_weight=weight,
            random_state=rs,
            n_jobs=-1,
        ),
    }


# ---------------------------------------------------------------------------
# Public training routine
# ---------------------------------------------------------------------------


def train_models(
    pre_X: np.ndarray,
    y: np.ndarray,
    cluster_labels: np.ndarray | None,
    cv,
    cfg: Config,
) -> Dict[str, object]:
    """Train tuned MLP and baseline models.

    Parameters
    ----------
    pre_X, y:
        Training data after preprocessing and GA feature selection.
    cluster_labels:
        Optional cluster assignments to be appended as one-hot features.
    cv:
        Cross-validation splitter.
    cfg:
        Configuration object with training options.
    """

    X = pre_X
    if cluster_labels is not None:
        labels = cluster_labels.astype(int)
        one_hot = np.eye(labels.max() + 1)[labels]
        X = np.hstack([X, one_hot])

    # Tune and train MLP --------------------------------------------------
    tune_result = pso_tune_mlp(X, y, cv, cfg)
    mlp = make_mlp(tune_result["params"], cfg)
    X_fit, y_fit, sample_weight = _handle_imbalance(X, y, cfg)
    mlp.fit(X_fit, y_fit, sample_weight=sample_weight)

    # Fit baselines ------------------------------------------------------
    baselines = _make_baselines(cfg)
    baseline_scores: Dict[str, float] = {}
    for name, model in baselines.items():
        if cfg.imbalance == "smote":
            estimator = ImbPipeline(
                steps=[("smote", SMOTE(random_state=cfg.cv.random_state)), ("clf", clone(model))]
            )
            score = float(
                cross_val_score(estimator, X, y, cv=cv, scoring="f1_weighted").mean()
            )
            model.fit(X_fit, y_fit)
        else:
            score = float(
                cross_val_score(model, X, y, cv=cv, scoring="f1_weighted").mean()
            )
            if cfg.imbalance == "class_weight":
                model.fit(X, y)
            else:
                model.fit(X, y)
        baseline_scores[name] = score

    # Persist models -----------------------------------------------------
    out_dir = Path(cfg.paths.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    # Persist tuned MLP under a simple and predictable name
    joblib.dump(mlp, out_dir / "mlp.joblib")
    for name, model in baselines.items():
        joblib.dump(model, out_dir / f"{name}.joblib")

    manifest = {
        "mlp_params": tune_result["params"],
        "mlp_best_cv_f1_weighted": tune_result["best_cv_f1_weighted"],
        "baseline_cv_f1_weighted": baseline_scores,
    }
    with (out_dir / "manifest.json").open("w") as fh:
        json.dump(manifest, fh, indent=2)

    return {
        "mlp": mlp,
        "baselines": baselines,
        "pso": tune_result,
        "cv_scores": baseline_scores,
    }


__all__ = ["build_model", "make_mlp", "train_models"]

