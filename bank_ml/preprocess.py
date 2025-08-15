"""Preprocessing utilities."""

from __future__ import annotations

from pathlib import Path
from typing import List, Tuple

import joblib
import numpy as np
import pandas as pd
from loguru import logger
from imblearn.over_sampling import SMOTE
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder


def train_test_split_data(
    X: pd.DataFrame, y: pd.Series, test_size: float, random_state: int
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    return train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )


def handle_imbalance(X: pd.DataFrame, y: pd.Series, method: str) -> Tuple[pd.DataFrame, pd.Series]:
    if method == "smote":
        sampler = SMOTE(random_state=0)
        return sampler.fit_resample(X, y)
    return X, y


# ---------------------------------------------------------------------------
# Column type detection and preprocessing pipelines
# ---------------------------------------------------------------------------

def _detect_column_types(X: pd.DataFrame) -> Tuple[List[str], List[str]]:
    """Return lists of numeric and categorical column names.

    Object, category and boolean dtypes are treated as categorical.
    """

    categorical_cols = X.select_dtypes(
        include=["object", "category", "bool"]
    ).columns.tolist()
    numeric_cols = [c for c in X.columns if c not in categorical_cols]
    return numeric_cols, categorical_cols


def build_preprocess(X: pd.DataFrame) -> Tuple[Pipeline, List[str]]:
    """Build a preprocessing pipeline for the given DataFrame.

    Parameters
    ----------
    X:
        Input feature matrix.

    Returns
    -------
    pipeline:
        An ``sklearn`` :class:`Pipeline` performing imputation, scaling and
        encoding.
    feature_names:
        Names of the output features produced by the pipeline.
    """

    numeric_cols, categorical_cols = _detect_column_types(X)
    logger.debug("Numeric columns detected: %s", numeric_cols)
    logger.debug("Categorical columns detected: %s", categorical_cols)

    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", MinMaxScaler()),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "encoder",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
            ),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, numeric_cols),
            ("cat", categorical_pipeline, categorical_cols),
        ]
    )

    pipeline = Pipeline([("preprocess", preprocessor)])

    # Fit a clone of the transformer to determine feature names without
    # mutating the returned pipeline
    preprocessor_clone = clone(preprocessor).fit(X)
    feature_names = list(preprocessor_clone.get_feature_names_out())

    return pipeline, feature_names


def fit_transform_preprocess(pipeline: Pipeline, X: pd.DataFrame) -> np.ndarray:
    """Fit the preprocessing pipeline and transform the data."""

    logger.debug("Fitting and transforming data with preprocessing pipeline")
    return pipeline.fit_transform(X)


def transform_preprocess(pipeline: Pipeline, X: pd.DataFrame) -> np.ndarray:
    """Transform new data with an already fitted pipeline."""

    logger.debug("Transforming data with fitted preprocessing pipeline")
    return pipeline.transform(X)


def save_preprocess(pipeline: Pipeline, path: Path | str) -> None:
    """Persist a fitted preprocessing pipeline to disk."""

    joblib.dump(pipeline, path)
    logger.info("Saved preprocessing pipeline to %s", path)


def load_preprocess(path: Path | str) -> Pipeline:
    """Load a previously persisted preprocessing pipeline."""

    pipeline = joblib.load(path)
    logger.info("Loaded preprocessing pipeline from %s", path)
    return pipeline
