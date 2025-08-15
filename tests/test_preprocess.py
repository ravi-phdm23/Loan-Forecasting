import numpy as np
import pandas as pd

from bank_ml.preprocess import (
    build_preprocess,
    fit_transform_preprocess,
    transform_preprocess,
)


def test_preprocess_pipeline_smoke():
    """Pipeline should fit and transform a tiny dataset."""

    X = pd.DataFrame(
        {
            "num": [1.0, 2.0, None, 4.0],
            "cat": ["a", "b", "a", None],
            "flag": [True, False, True, False],
        }
    )

    pipeline, feature_names = build_preprocess(X)
    Xt = fit_transform_preprocess(pipeline, X)

    assert Xt.shape[0] == X.shape[0]
    assert Xt.shape[1] == len(feature_names)

    Xt2 = transform_preprocess(pipeline, X)
    np.testing.assert_allclose(Xt, Xt2)

