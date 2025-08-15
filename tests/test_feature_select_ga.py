from pathlib import Path

import numpy as np
from sklearn.model_selection import StratifiedKFold

from bank_ml.config import Config, Paths, GA
from bank_ml.feature_select_ga import ga_select_features


def test_ga_select_features_smoke(tmp_path: Path):
    rng = np.random.default_rng(0)
    X = rng.normal(size=(20, 12))
    y = np.array([0, 1] * 10)
    feature_names = [f"f{i}" for i in range(X.shape[1])]

    cv = StratifiedKFold(n_splits=2, shuffle=True, random_state=0)
    cfg = Config(
        paths=Paths(input_csv=Path("in.csv"), output_dir=tmp_path),
        label="y",
        ga=GA(pop=4, gens=2, cx_prob=0.5, mut_prob=0.5),
        imbalance="none",
    )

    result = ga_select_features(X, y, feature_names, cv, cfg)

    mask = result["mask"]
    assert mask.any()
    assert len(result["selected_names"]) == int(mask.sum())

