"""Clustering utilities (placeholder)."""
from __future__ import annotations
from typing import Tuple

import numpy as np
from sklearn.cluster import KMeans


def choose_k(X: np.ndarray, k_grid: list[int]) -> int:
    """Select the best *k* based on inertia (placeholder heuristic)."""
    inertia = {}
    for k in k_grid:
        model = KMeans(n_clusters=k, n_init=10, random_state=0)
        model.fit(X)
        inertia[k] = model.inertia_
    return min(inertia, key=inertia.get)


def cluster_data(X: np.ndarray, k: int) -> Tuple[KMeans, np.ndarray]:
    model = KMeans(n_clusters=k, n_init=10, random_state=0)
    labels = model.fit_predict(X)
    return model, labels
