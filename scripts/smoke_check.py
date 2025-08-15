from __future__ import annotations

"""Lightweight end-to-end verification of the Bank ML pipeline."""

import json
from pathlib import Path
import tempfile

import joblib
import numpy as np
import pandas as pd
import yaml

from bank_ml import clustering, preprocess
from bank_ml.cli import fit


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    # ------------------------------------------------------------------
    # Prepare small dataset and temporary config
    # ------------------------------------------------------------------
    df = pd.read_csv(ROOT / "banking_dataset.csv").head(50)
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        data_path = tmp_path / "sample.csv"
        df.to_csv(data_path, index=False)

        cfg = {
            "paths": {
                "input_csv": str(data_path),
                "output_dir": str(tmp_path / "outputs"),
            },
            "label": "Label_LoanDefault",
            "id_column": "CustomerID",
            "cv": {"n_splits": 2, "test_size": 0.2, "random_state": 42},
            "imbalance": {"method": "class_weight"},
            "ga": {"pop": 4, "gens": 2, "cx_prob": 0.8, "mut_prob": 0.1},
            "clustering": {"k_grid": [2], "n_init": 1, "max_iter": 10},
            "pso": {"particles": 4, "iters": 2},
        }
        cfg_path = tmp_path / "config.yaml"
        with cfg_path.open("w") as fh:
            yaml.safe_dump(cfg, fh)

        # ------------------------------------------------------------------
        # Run training pipeline in fast mode
        # ------------------------------------------------------------------
        fit(config=cfg_path, fast=True)

        out_root = tmp_path / "outputs"
        run_dirs = sorted(out_root.iterdir())
        assert run_dirs, "No output produced"
        run_dir = run_dirs[-1]

        # ------------------------------------------------------------------
        # Verify expected artefacts
        # ------------------------------------------------------------------
        metrics_path = run_dir / "metrics.csv"
        model_path = run_dir / "mlp.joblib"
        assert metrics_path.exists(), "metrics.csv missing"
        assert model_path.exists(), "mlp.joblib missing"

        # ------------------------------------------------------------------
        # Load artefacts and generate predictions for 5 rows
        # ------------------------------------------------------------------
        preproc = preprocess.load_preprocess(run_dir / "preprocess.joblib")
        ga_info = json.load((run_dir / "ga_selection.json").open())
        mask = np.array(ga_info["mask"], dtype=bool)
        cluster_model = joblib.load(run_dir / "clustering.joblib")
        model = joblib.load(model_path)

        X = df.drop(columns=["Label_LoanDefault", "CustomerID"]).head(5)
        X_pre = preprocess.transform_preprocess(preproc, X)
        X_sel = X_pre[:, mask]
        labels = cluster_model.predict(X_sel)
        X_aug = clustering.append_cluster_features(X_sel, labels)
        preds = model.predict(X_aug)
        assert len(preds) == 5

        print(f"Smoke check passed. Artefacts saved under: {run_dir}")


if __name__ == "__main__":
    main()
