"""Particle swarm optimisation for :class:`~sklearn.neural_network.MLPClassifier`.

The original project only provided a tiny placeholder.  This module now
contains a fully fledged PSO based hyper–parameter search that can operate
either with the :mod:`pyswarms` library (when available) or with a small
NumPy implementation used as a fallback.

The public entry point is :func:`pso_tune_mlp` which performs the search and
returns the best parameters together with a convergence trajectory.  The
search space is inferred from :class:`bank_ml.config.Config`.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import f1_score
from sklearn.model_selection import check_cv
from sklearn.neural_network import MLPClassifier
from sklearn.utils.class_weight import compute_class_weight

try:  # pragma: no cover - optional dependency
    import pyswarms as ps  # type: ignore
except Exception:  # pragma: no cover - pyswarms is optional
    ps = None

from imblearn.over_sampling import SMOTE

from .config import Config


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------


def _decode_particle(
    particle: np.ndarray, activation_choices: List[str]
) -> Dict[str, object]:
    """Convert a particle position to MLP hyper-parameters.

    Parameters
    ----------
    particle:
        Array representing a particle position in the search space.
    activation_choices:
        Ordered list of activation functions.  The last element of ``particle``
        encodes the index into this list.
    """

    hidden1 = int(round(particle[0]))
    hidden2 = int(round(particle[1]))
    lr = 10 ** particle[2]
    alpha = 10 ** particle[3]
    momentum = float(particle[4])

    act_idx = int(round(particle[5]))
    act_idx = int(np.clip(act_idx, 0, len(activation_choices) - 1))
    activation = activation_choices[act_idx]

    return {
        "hidden_layer_sizes": (hidden1, hidden2),
        "learning_rate_init": lr,
        "alpha": alpha,
        "momentum": momentum,
        "activation": activation,
    }


def _mean_f1_score(
    params: Dict[str, object],
    X: np.ndarray,
    y: np.ndarray,
    cv,
    imbalance: str,
) -> float:
    """Compute mean weighted F1-score using cross validation."""

    cv = check_cv(cv, y, classifier=True)
    scores: List[float] = []

    for train_idx, test_idx in cv.split(X, y):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        # Handle imbalance according to configuration
        sample_weight = None
        if imbalance == "smote":
            sampler = SMOTE(random_state=0)
            X_train, y_train = sampler.fit_resample(X_train, y_train)
        elif imbalance == "class_weight":
            classes = np.unique(y_train)
            weights = compute_class_weight("balanced", classes=classes, y=y_train)
            weight_map = dict(zip(classes, weights))
            sample_weight = np.array([weight_map[c] for c in y_train])

        clf = MLPClassifier(
            **params,
            solver="sgd",
            early_stopping=True,
            max_iter=300,
            random_state=0,
        )

        clf.fit(X_train, y_train, sample_weight=sample_weight)
        pred = clf.predict(X_test)
        scores.append(f1_score(y_test, pred, average="weighted"))

    return float(np.mean(scores))


# ---------------------------------------------------------------------------
# Minimal PSO implementation -------------------------------------------------
# ---------------------------------------------------------------------------


def _basic_pso(
    func: Callable[[np.ndarray], float],
    low: np.ndarray,
    high: np.ndarray,
    particles: int,
    iters: int,
) -> Tuple[np.ndarray, List[float]]:
    """A very small PSO implementation with dynamic coefficients.

    Parameters follow the usual PSO notation.  ``func`` should return a cost to
    *minimise* for a single particle.
    """

    dim = len(low)
    pos = low + np.random.rand(particles, dim) * (high - low)
    vel = np.zeros((particles, dim))

    pbest = pos.copy()
    pbest_score = np.array([func(p) for p in pos])
    gbest_idx = np.argmin(pbest_score)
    gbest = pbest[gbest_idx].copy()
    gbest_score = pbest_score[gbest_idx]

    trajectory: List[float] = []

    for t in range(iters):
        # Linearly varying parameters
        frac = t / max(iters - 1, 1)
        w = 0.9 - 0.5 * frac
        c1 = 2.0 - frac
        c2 = 2.0 + frac

        r1 = np.random.rand(particles, dim)
        r2 = np.random.rand(particles, dim)
        vel = w * vel + c1 * r1 * (pbest - pos) + c2 * r2 * (gbest - pos)
        pos = np.clip(pos + vel, low, high)

        scores = np.array([func(p) for p in pos])

        improve = scores < pbest_score
        pbest[improve] = pos[improve]
        pbest_score[improve] = scores[improve]

        best_idx = np.argmin(pbest_score)
        if pbest_score[best_idx] < gbest_score:
            gbest_score = pbest_score[best_idx]
            gbest = pbest[best_idx].copy()

        trajectory.append(-gbest_score)

    return gbest, trajectory


# ---------------------------------------------------------------------------
# Public API ----------------------------------------------------------------
# ---------------------------------------------------------------------------


def pso_tune_mlp(
    X: np.ndarray, y: np.ndarray, cv, cfg: Config
) -> Dict[str, object]:
    """Perform PSO hyper-parameter search for ``MLPClassifier``.

    Parameters
    ----------
    X, y:
        Training data.
    cv:
        Any scikit-learn compatible cross-validation splitter.
    cfg:
        Configuration object containing PSO and search space settings.

    Returns
    -------
    dict
        ``{"params": best_params, "best_cv_f1_weighted": score, "trajectory": history}``
    """

    bounds = cfg.mlp_bounds
    activation_choices = list(bounds.activation_choices)

    low = np.array(
        [
            bounds.hidden1[0],
            bounds.hidden2[0],
            np.log10(bounds.lr[0]),
            np.log10(bounds.alpha[0]),
            bounds.momentum[0],
            0,
        ]
    )
    high = np.array(
        [
            bounds.hidden1[1],
            bounds.hidden2[1],
            np.log10(bounds.lr[1]),
            np.log10(bounds.alpha[1]),
            bounds.momentum[1],
            len(activation_choices) - 1,
        ]
    )

    def cost_single(particle: np.ndarray) -> float:
        params = _decode_particle(particle, activation_choices)
        score = _mean_f1_score(params, X, y, cv, cfg.imbalance)
        return -score  # minimise

    particles = cfg.pso.particles
    iters = cfg.pso.iters

    if ps is not None:  # pragma: no cover - requires optional dependency
        def batch_cost(pos: np.ndarray) -> np.ndarray:
            return np.array([cost_single(p) for p in pos])

        optimizer = ps.single.GlobalBestPSO(
            n_particles=particles,
            dimensions=len(low),
            bounds=(low, high),
            options={"c1": 1.5, "c2": 1.5, "w": 0.7},
        )
        best_cost, best_pos = optimizer.optimize(batch_cost, iters)
        trajectory = [-c for c in optimizer.cost_history]
    else:
        best_pos, trajectory = _basic_pso(cost_single, low, high, particles, iters)
        best_cost = -max(trajectory)

    best_params = _decode_particle(best_pos, activation_choices)
    best_score = -best_cost

    # ------------------------------------------------------------------
    # Save convergence plot
    # ------------------------------------------------------------------
    assets_dir = Path(__file__).resolve().parent.parent / "assets"
    assets_dir.mkdir(parents=True, exist_ok=True)
    plot_path = assets_dir / "pso_convergence.png"
    plt.figure()
    plt.plot(range(1, len(trajectory) + 1), trajectory)
    plt.xlabel("Iteration")
    plt.ylabel("Best F1-weighted")
    plt.title("PSO convergence")
    plt.tight_layout()
    plt.savefig(plot_path)
    plt.close()

    return {
        "params": best_params,
        "best_cv_f1_weighted": best_score,
        "trajectory": trajectory,
    }


__all__ = ["pso_tune_mlp"]

