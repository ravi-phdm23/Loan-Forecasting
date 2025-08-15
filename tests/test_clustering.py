"""Tests for clustering utilities."""

from pathlib import Path

import numpy as np
import pytest
from sklearn.datasets import make_blobs

from bank_ml.clustering import append_cluster_features, select_k_and_cluster
from bank_ml.config import CV, Clustering, Config, Paths


def _dummy_config(tmp_path: Path) -> Config:
    paths = Paths(input_csv=tmp_path / "dummy.csv", output_dir=tmp_path)
    return Config(paths=paths, label="label", cv=CV(random_state=0), clustering=Clustering())


def test_select_k_and_cluster_basic(tmp_path):
    X, _ = make_blobs(n_samples=90, centers=3, random_state=0)
    cfg = _dummy_config(tmp_path)

    res = select_k_and_cluster(X, [2, 3, 4], cfg)

    assert res["k"] == 3
    assert res["labels"].shape[0] == X.shape[0]
    assert res["centers"].shape == (3, X.shape[1])

    assert Path("assets/dbi_per_k.png").is_file()
    assert Path("assets/sil_per_k.png").is_file()


def test_select_k_prefers_three_when_close(tmp_path, monkeypatch):
    X = np.random.RandomState(0).randn(50, 2)
    cfg = _dummy_config(tmp_path)

    dbi_values = {2: 0.5, 3: 0.51, 4: 0.7}
    sil_values = {2: 0.6, 3: 0.59, 4: 0.4}

    def fake_dbi(X, labels):
        k = len(np.unique(labels))
        return dbi_values[k]

    def fake_sil(X, labels):
        k = len(np.unique(labels))
        return sil_values[k]

    monkeypatch.setattr("bank_ml.clustering.davies_bouldin_score", fake_dbi)
    monkeypatch.setattr("bank_ml.clustering.silhouette_score", fake_sil)

    res = select_k_and_cluster(X, [2, 3, 4], cfg)
    assert res["k"] == 3


def test_append_cluster_features():
    X = np.array([[1.0, 2.0], [3.0, 4.0]])
    labels = np.array([0, 1])
    X_aug = append_cluster_features(X, labels)

    assert X_aug.shape == (2, 4)
    assert np.array_equal(X_aug[:, :2], X)
    assert np.array_equal(X_aug[:, 2:], np.eye(2))

