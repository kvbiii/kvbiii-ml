"""Additional tests for CrossFeatureGenerator (cross_encoding)."""

import numpy as np
import pandas as pd
import pytest

from kvbiii_ml.data_processing.feature_engineering.cross_encoding import (
    CrossFeatureGenerator,
)


@pytest.fixture
def cross_data():
    return pd.DataFrame(
        {
            "A": ["a", "b", "a", "c"],
            "B": ["x", "y", "x", "z"],
            "C": [1, 2, 1, 3],
            "D": [10, 20, 10, 30],
        }
    )


def test_init_validation_errors():
    with pytest.raises(ValueError):
        CrossFeatureGenerator(features_names="not_a_list")  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        CrossFeatureGenerator(features_names=["A"], degree=3)
    with pytest.raises(ValueError):
        CrossFeatureGenerator(features_names=["A"], degree=1)


def test_fit_transform_generates_expected_columns(cross_data):
    gen = CrossFeatureGenerator(features_names=["A", "B", "C", "D"], degree=2)
    out = gen.fit_transform(cross_data)
    # Number of new columns equals combinations C(4,2)=6
    assert len(gen.get_feature_names()) == 6
    for col in gen.get_feature_names():
        assert col in out.columns


def test_transform_new_data_handles_unseen_categories(cross_data):
    gen = CrossFeatureGenerator(features_names=["A", "B"], degree=2)
    gen.fit(cross_data)
    new_df = pd.DataFrame({"A": ["q"], "B": ["x"]})
    transformed = gen.transform(new_df)
    combo_name = "-".join(sorted(["A", "B"]))
    assert combo_name in transformed.columns
    # Unseen combination at least maps to -1
    assert transformed[combo_name].iloc[0] in (-1, 0, 1)


def test_numerical_interaction(cross_data):
    gen = CrossFeatureGenerator(features_names=["C", "D"], degree=2)
    out = gen.fit_transform(cross_data)
    combo = "-".join(sorted(["C", "D"]))
    assert combo in out.columns
    expected = (cross_data["C"] * cross_data["D"]).values
    assert np.array_equal(out[combo].values, expected)


if __name__ == "__main__":
    print("Run this file with pytest to execute tests.")
