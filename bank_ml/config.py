"""Configuration management for bank_ml."""
from __future__ import annotations
from pathlib import Path
from typing import List, Optional

import yaml
from pydantic import BaseModel, Field


class Paths(BaseModel):
    input_csv: Path
    output_dir: Path


class CV(BaseModel):
    n_splits: int = 10
    test_size: float = 0.3
    random_state: int = 42


class GA(BaseModel):
    pop: int = 30
    gens: int = 25
    cx_prob: float = Field(0.8, alias="cx_prob")
    mut_prob: float = Field(0.1, alias="mut_prob")


class Clustering(BaseModel):
    k_grid: List[int] = Field(default_factory=lambda: [2, 3, 4, 5, 6])
    n_init: int = 20
    max_iter: int = 300


class PSO(BaseModel):
    particles: int = 24
    iters: int = 30


class MLPBounds(BaseModel):
    hidden1: List[int] = Field(default_factory=lambda: [16, 64])
    hidden2: List[int] = Field(default_factory=lambda: [16, 64])
    lr: List[float] = Field(default_factory=lambda: [1e-4, 1e-1])
    alpha: List[float] = Field(default_factory=lambda: [1e-6, 1e-2])
    momentum: List[float] = Field(default_factory=lambda: [0.5, 0.95])
    activation_choices: List[str] = Field(default_factory=lambda: ["logistic", "tanh", "relu"])


class Config(BaseModel):
    paths: Paths
    label: str
    id_column: Optional[str] = None
    cv: CV = CV()
    imbalance: str = Field(
        "class_weight", pattern="^(none|class_weight|smote)$"
    )
    ga: GA = GA()
    clustering: Clustering = Clustering()
    pso: PSO = PSO()
    mlp_bounds: MLPBounds = MLPBounds()


def load_config(path: Path | str) -> Config:
    """Load a configuration from a YAML file."""
    path = Path(path)
    with path.open("r") as fh:
        data = yaml.safe_load(fh) or {}

    # Allow ``imbalance`` to be specified either as a plain string or as a
    # mapping with a ``method`` field.  The README examples use the latter form
    # for readability, so we normalise it here for the ``Config`` model.
    imb = data.get("imbalance")
    if isinstance(imb, dict) and "method" in imb:
        data["imbalance"] = imb["method"]

    return Config.model_validate(data)


def ensure_output_dirs(config: Config) -> Path:
    """Ensure that the output directory exists.

    Returns the path to the output directory.
    """
    out_path = config.paths.output_dir
    out_path.mkdir(parents=True, exist_ok=True)
    return out_path
