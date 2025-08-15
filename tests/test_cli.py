from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from typer.testing import CliRunner

from bank_ml.cli import app


def write_config(tmp_path: Path) -> Path:
    rng = np.random.RandomState(0)
    data = pd.DataFrame(
        {
            "f1": rng.randn(50),
            "f2": rng.randn(50),
            "f3": rng.randn(50),
            "label": rng.randint(0, 2, 50),
        }
    )
    data_path = tmp_path / "data.csv"
    data.to_csv(data_path, index=False)
    cfg = {
        "paths": {"input_csv": str(data_path), "output_dir": str(tmp_path / "out")},
        "label": "label",
        "cv": {"n_splits": 2, "test_size": 0.2, "random_state": 0},
        "ga": {"pop": 4, "gens": 1, "cx_prob": 0.5, "mut_prob": 0.5},
        "clustering": {"k_grid": [2], "n_init": 1, "max_iter": 50},
        "pso": {"particles": 2, "iters": 1},
        "mlp_bounds": {
            "hidden1": [2, 4],
            "hidden2": [2, 4],
            "lr": [1e-3, 1e-2],
            "alpha": [1e-5, 1e-4],
            "momentum": [0.1, 0.9],
            "activation_choices": ["relu", "tanh"],
        },
        "imbalance": "none",
    }
    cfg_path = tmp_path / "config.yaml"
    with cfg_path.open("w") as fh:
        yaml.safe_dump(cfg, fh)
    return cfg_path


def test_cli_fit_generates_metrics(tmp_path: Path):
    cfg_path = write_config(tmp_path)
    runner = CliRunner()
    result = runner.invoke(app, ["fit", "--config", str(cfg_path)])
    assert result.exit_code == 0
    out_subdirs = list((tmp_path / "out").glob("*"))
    assert out_subdirs, "no output directory created"
    metrics_path = out_subdirs[0] / "metrics.csv"
    assert metrics_path.exists()
