"""Particle swarm optimization utilities (placeholder)."""
from __future__ import annotations
from typing import Callable, List

import numpy as np

try:  # pragma: no cover - optional dependency
    import pyswarms as ps  # type: ignore
except Exception:  # pragma: no cover
    ps = None


def optimize(func: Callable[[np.ndarray], np.ndarray], bounds: List[tuple[float, float]], particles: int, iters: int) -> np.ndarray:
    """Very small PSO wrapper with NumPy fallback."""
    low = np.array([b[0] for b in bounds])
    high = np.array([b[1] for b in bounds])
    if ps is None:
        best = low
        best_score = func(best)
        for _ in range(iters * particles):
            candidate = low + np.random.rand(len(bounds)) * (high - low)
            score = func(candidate)
            if score < best_score:
                best, best_score = candidate, score
        return best
    else:  # pragma: no cover
        optimizer = ps.single.GlobalBestPSO(n_particles=particles, dimensions=len(bounds), options={"c1": 0.5, "c2": 0.3, "w": 0.9}, bounds=(low, high))
        cost, pos = optimizer.optimize(lambda x: func(x).mean(axis=1), iters)
        return pos
