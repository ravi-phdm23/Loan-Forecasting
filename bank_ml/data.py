"""Data loading utilities."""
from pathlib import Path
from typing import Optional, Tuple

import pandas as pd


def load_csv(path: Path | str, label: str, id_column: Optional[str] = None) -> Tuple[pd.DataFrame, pd.Series]:
    """Load a CSV file and split into features and target."""
    df = pd.read_csv(path)
    y = df[label]
    X = df.drop(columns=[label])
    if id_column and id_column in X.columns:
        X = X.drop(columns=[id_column])
    return X, y
