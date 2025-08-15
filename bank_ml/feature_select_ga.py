"""Genetic algorithm based feature selection."""

from __future__ import annotations

from pathlib import Path
from typing import List, Dict

import numpy as np

from bank_ml.config import Config

try:  # pragma: no cover - optional dependency
    from deap import base, creator, tools  # type: ignore
except Exception:  # pragma: no cover
    base = creator = tools = None

try:  # pragma: no cover - optional dependency
    import matplotlib.pyplot as plt  # type: ignore
except Exception:  # pragma: no cover
    plt = None

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score


def _ensure_min_features(mask: np.ndarray, min_selected: int) -> None:
    """Ensure that a boolean mask selects at least ``min_selected`` features."""

    if mask.sum() < min_selected:
        idx = np.random.choice(mask.size, min_selected, replace=False)
        mask[idx] = True


def _plot_history(history: List[float], out_file: Path) -> None:
    """Plot the best score per generation."""

    if plt is None:  # pragma: no cover - plotting is optional
        return
    out_file.parent.mkdir(parents=True, exist_ok=True)
    plt.figure()
    plt.plot(range(1, len(history) + 1), history)
    plt.xlabel("Generation")
    plt.ylabel("Best F1-weighted")
    plt.title("GA Feature Selection")
    plt.tight_layout()
    plt.savefig(out_file)
    plt.close()


def _eval_mask(
    X_np: np.ndarray,
    y: np.ndarray,
    mask: np.ndarray,
    cv,
    class_weight,
    min_selected: int,
) -> float:
    if mask.sum() < min_selected:
        return 0.0
    X_sel = X_np[:, mask]
    if X_sel.shape[1] == 0:
        return 0.0
    model = LogisticRegression(max_iter=200, class_weight=class_weight)
    scores = cross_val_score(model, X_sel, y, cv=cv, scoring="f1_weighted")
    return float(scores.mean())


def _numpy_ga(
    X_np: np.ndarray,
    y: np.ndarray,
    cv,
    cfg: Config,
    min_selected: int,
    class_weight,
) -> tuple[np.ndarray, List[float], float]:
    """Simple GA using NumPy operations."""

    rng = np.random.default_rng()
    n_features = X_np.shape[1]
    pop_size = cfg.ga.pop
    gens = cfg.ga.gens
    cx_prob = cfg.ga.cx_prob
    mut_prob = cfg.ga.mut_prob

    def fitness(mask: np.ndarray) -> float:
        return _eval_mask(X_np, y, mask, cv, class_weight, min_selected)

    def init_individual() -> np.ndarray:
        mask = rng.random(n_features) < 0.5
        _ensure_min_features(mask, min_selected)
        return mask

    population = np.array([init_individual() for _ in range(pop_size)], dtype=bool)
    scores = np.array([fitness(ind) for ind in population])

    history: List[float] = []
    best_mask = population[np.argmax(scores)].copy()
    best_score = float(scores.max())

    for _ in range(gens):
        # selection (tournament of size 3)
        selected = []
        for _ in range(pop_size):
            idx = rng.integers(pop_size, size=3)
            best = population[idx[np.argmax(scores[idx])]].copy()
            selected.append(best)
        population = np.array(selected, dtype=bool)

        # crossover (one-point)
        for i in range(0, pop_size, 2):
            if i + 1 >= pop_size:
                break
            if rng.random() < cx_prob:
                cx_point = rng.integers(1, n_features)
                tmp = population[i, cx_point:].copy()
                population[i, cx_point:] = population[i + 1, cx_point:]
                population[i + 1, cx_point:] = tmp

        # mutation (bit flip)
        for i in range(pop_size):
            if rng.random() < mut_prob:
                flips = rng.random(n_features) < (1.0 / n_features)
                population[i] = np.logical_xor(population[i], flips)
            _ensure_min_features(population[i], min_selected)

        scores = np.array([fitness(ind) for ind in population])
        gen_best_idx = int(np.argmax(scores))
        gen_best = float(scores[gen_best_idx])
        history.append(gen_best)
        if gen_best > best_score:
            best_score = gen_best
            best_mask = population[gen_best_idx].copy()

    return best_mask, history, best_score


def ga_select_features(
    X_np: np.ndarray,
    y: np.ndarray,
    feature_names: List[str],
    cv,
    cfg: Config,
) -> Dict[str, object]:
    """Run a GA to select features.

    Parameters
    ----------
    X_np, y
        Preprocessed feature matrix and labels.
    feature_names
        Names corresponding to columns of ``X_np``.
    cv
        A scikit-learn cross-validation splitter.
    cfg
        Configuration containing GA parameters.
    """

    n_features = X_np.shape[1]
    min_selected = 5 if n_features >= 10 else 1
    class_weight = "balanced" if cfg.imbalance != "none" else None

    if tools is not None:  # pragma: no cover - exercised if deap is installed
        import random

        creator.create("FitnessMax", base.Fitness, weights=(1.0,))
        creator.create("Individual", list, fitness=creator.FitnessMax)

        toolbox = base.Toolbox()
        toolbox.register("attr_bool", random.randint, 0, 1)
        toolbox.register(
            "individual", tools.initRepeat, creator.Individual, toolbox.attr_bool, n_features
        )
        toolbox.register("population", tools.initRepeat, list, toolbox.individual)

        def evaluate(individual: list[int]) -> tuple[float]:
            mask = np.array(individual, dtype=bool)
            score = _eval_mask(X_np, y, mask, cv, class_weight, min_selected)
            return (score,)

        toolbox.register("mate", tools.cxOnePoint)
        toolbox.register("mutate", tools.mutFlipBit, indpb=1.0 / n_features)
        toolbox.register("select", tools.selTournament, tournsize=3)
        toolbox.register("evaluate", evaluate)

        population = toolbox.population(n=cfg.ga.pop)
        for ind in population:
            mask = np.array(ind, dtype=bool)
            _ensure_min_features(mask, min_selected)
            for i, val in enumerate(mask):
                ind[i] = int(val)

        history: List[float] = []
        best_ind = None
        best_score = -1.0

        # evaluate initial population
        fitnesses = list(map(toolbox.evaluate, population))
        for ind, fit in zip(population, fitnesses):
            ind.fitness.values = fit

        for _ in range(cfg.ga.gens):
            offspring = toolbox.select(population, len(population))
            offspring = list(map(toolbox.clone, offspring))

            for child1, child2 in zip(offspring[::2], offspring[1::2]):
                if random.random() < cfg.ga.cx_prob:
                    toolbox.mate(child1, child2)
                    del child1.fitness.values
                    del child2.fitness.values

            for mutant in offspring:
                if random.random() < cfg.ga.mut_prob:
                    toolbox.mutate(mutant)
                    del mutant.fitness.values
                mask = np.array(mutant, dtype=bool)
                _ensure_min_features(mask, min_selected)
                for i, val in enumerate(mask):
                    mutant[i] = int(val)

            invalid = [ind for ind in offspring if not ind.fitness.valid]
            fitnesses = list(map(toolbox.evaluate, invalid))
            for ind, fit in zip(invalid, fitnesses):
                ind.fitness.values = fit

            population[:] = offspring

            fits = [ind.fitness.values[0] for ind in population]
            gen_best = max(fits)
            history.append(gen_best)
            if gen_best > best_score:
                best_score = gen_best
                best_ind = tools.selBest(population, 1)[0]

        if best_ind is None:
            best_ind = tools.selBest(population, 1)[0]
            best_score = best_ind.fitness.values[0]

        best_mask = np.array(best_ind, dtype=bool)

    else:
        best_mask, history, best_score = _numpy_ga(
            X_np, y, cv, cfg, min_selected, class_weight
        )

    selected_names = [name for name, flag in zip(feature_names, best_mask) if flag]

    _plot_history(history, Path("assets") / "ga_history.png")

    return {
        "mask": best_mask,
        "selected_names": selected_names,
        "cv_f1_weighted": float(best_score),
        "history": history,
    }


__all__ = ["ga_select_features"]

