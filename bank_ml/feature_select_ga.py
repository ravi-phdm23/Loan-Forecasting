"""Genetic algorithm based feature selection (placeholder)."""
from __future__ import annotations
from typing import List

import numpy as np

try:  # pragma: no cover - optional dependency
    from deap import base, creator, tools  # type: ignore
except Exception:  # pragma: no cover
    base = creator = tools = None


def run_ga(feature_count: int) -> List[int]:
    """Dummy GA that selects half of the features.

    If :mod:`deap` is available, a real GA implementation could be placed here.
    """
    idx = np.arange(feature_count)
    np.random.shuffle(idx)
    return sorted(idx[: max(1, feature_count // 2)])
