"""Clustering utilities for GA-selected features.

This module provides functionality to select an appropriate number of
clusters using internal clustering validation indices and to append
cluster indicators to feature matrices.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict

import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import davies_bouldin_score, silhouette_score

from .config import Config


def _plot_metric(k_vals: list[int], metric: Dict[int, float], ylabel: str, filename: str) -> None:
    """Plot a metric versus ``k`` and save under the assets directory."""

    asset_dir = Path("assets")
    asset_dir.mkdir(exist_ok=True)
    plt.figure()
    plt.plot(k_vals, [metric[k] for k in k_vals], marker="o")
    plt.xlabel("k")
    plt.ylabel(ylabel)
    plt.xticks(k_vals)
    plt.tight_layout()
    plt.savefig(asset_dir / filename)
    plt.close()


def select_k_and_cluster(
    X_selected: np.ndarray, k_grid: list[int], cfg: Config
) -> dict:
    """Select ``k`` using DBI and Silhouette and perform clustering.

    Parameters
    ----------
    X_selected:
        Feature matrix after GA selection.
    k_grid:
        Candidate numbers of clusters.
    cfg:
        Configuration providing clustering parameters.

    Returns
    -------
    dict
        Dictionary containing the chosen ``k``, labels, per-k metrics and
        cluster centres.
    """

    dbi_per_k: Dict[int, float] = {}
    sil_per_k: Dict[int, float] = {}
    models: Dict[int, KMeans] = {}
    labels_per_k: Dict[int, np.ndarray] = {}

    for k in k_grid:
        model = KMeans(
            n_clusters=k,
            n_init=cfg.clustering.n_init,
            max_iter=cfg.clustering.max_iter,
            random_state=cfg.cv.random_state,
        )
        labels = model.fit_predict(X_selected)
        dbi = davies_bouldin_score(X_selected, labels)
        sil = silhouette_score(X_selected, labels)

        dbi_per_k[k] = float(dbi)
        sil_per_k[k] = float(sil)
        models[k] = model
        labels_per_k[k] = labels

    # Determine best k: minimal DBI, tie-broken by maximal Silhouette
    sorted_k = sorted(k_grid, key=lambda k: (dbi_per_k[k], -sil_per_k[k]))
    best_k = sorted_k[0]
    best_dbi = dbi_per_k[best_k]
    best_sil = sil_per_k[best_k]

    # If metrics are within 5% of those for k=3, prefer k=3
    if 3 in k_grid:
        dbi3 = dbi_per_k[3]
        sil3 = sil_per_k[3]
        if (
            abs(dbi3 - best_dbi) / best_dbi <= 0.05
            and abs(best_sil - sil3) / max(best_sil, 1e-12) <= 0.05
        ):
            best_k = 3
            best_dbi = dbi3
            best_sil = sil3

    _plot_metric(k_grid, dbi_per_k, "Davies-Bouldin Index", "dbi_per_k.png")
    _plot_metric(k_grid, sil_per_k, "Silhouette Score", "sil_per_k.png")

    model = models[best_k]
    labels = labels_per_k[best_k]

    return {
        "k": best_k,
        "labels": labels.astype(int),
        "dbi_per_k": dbi_per_k,
        "sil_per_k": sil_per_k,
        "centers": model.cluster_centers_,
        "model": model,
    }


def append_cluster_features(X: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """Append one-hot encoded cluster labels to ``X``.

    Parameters
    ----------
    X:
        Original feature matrix.
    labels:
        Cluster labels from :func:`select_k_and_cluster`.

    Returns
    -------
    np.ndarray
        Augmented feature matrix including cluster indicators.
    """

    labels = labels.astype(int)
    n_clusters = labels.max() + 1
    one_hot = np.eye(n_clusters)[labels]
    return np.hstack([X, one_hot])


__all__ = ["select_k_and_cluster", "append_cluster_features"]

