"""Preprocessing utilities."""
from typing import Tuple

import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split


def train_test_split_data(X: pd.DataFrame, y: pd.Series, test_size: float, random_state: int) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    return train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)


def handle_imbalance(X: pd.DataFrame, y: pd.Series, method: str) -> Tuple[pd.DataFrame, pd.Series]:
    if method == "smote":
        sampler = SMOTE(random_state=0)
        return sampler.fit_resample(X, y)
    return X, y
