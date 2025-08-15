"""Data loading utilities."""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

import pandas as pd
from loguru import logger
from sklearn.model_selection import StratifiedKFold, train_test_split

from .config import Config


def load_csv(
    path: Path | str, label: str, id_column: Optional[str] = None
) -> Tuple[pd.DataFrame, pd.Series]:
    """Load a CSV file and split into features and target.

    Parameters
    ----------
    path:
        Path to the CSV file.
    label:
        Name of the target column.
    id_column:
        Optional column containing row identifiers. If provided, the column is
        used as the index of the returned data but excluded from the feature
        matrix.
    """

    path = Path(path)
    df = pd.read_csv(path)
    if label not in df.columns:
        raise ValueError(f"Label column '{label}' not found in {path}")

    y = df[label]
    X = df.drop(columns=[label])

    if id_column and id_column in df.columns:
        ids = df[id_column]
        X = X.drop(columns=[id_column])
        X.index = ids
        y.index = ids

    logger.debug(
        "Loaded CSV %s with %d rows and %d columns", path, X.shape[0], X.shape[1] + 1
    )
    return X, y


def load_dataset(cfg: Config) -> Tuple[pd.DataFrame, pd.Series]:
    """Load the dataset defined by a :class:`Config` object.

    Parameters
    ----------
    cfg:
        Configuration object specifying paths and column names.
    """

    return load_csv(cfg.paths.input_csv, cfg.label, cfg.id_column)


def train_test_cv_split(
    X: pd.DataFrame, y: pd.Series, cfg: Config
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, StratifiedKFold]:
    """Perform a train/test split and build a cross-validation splitter.

    Returns the training and test partitions along with a ``StratifiedKFold``
    object configured according to ``cfg.cv``.
    """

    logger.debug(
        "Splitting data: test_size=%s, random_state=%s", cfg.cv.test_size, cfg.cv.random_state
    )
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=cfg.cv.test_size,
        random_state=cfg.cv.random_state,
        stratify=y,
    )

    cv = StratifiedKFold(
        n_splits=cfg.cv.n_splits,
        shuffle=True,
        random_state=cfg.cv.random_state,
    )
    return X_train, X_test, y_train, y_test, cv
