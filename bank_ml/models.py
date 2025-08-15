"""Model building utilities."""
from __future__ import annotations

from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def build_model(imbalance: str) -> Pipeline:
    class_weight = "balanced" if imbalance == "class_weight" else None
    clf = LogisticRegression(max_iter=200, class_weight=class_weight)
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("model", clf),
    ])
    return pipe
