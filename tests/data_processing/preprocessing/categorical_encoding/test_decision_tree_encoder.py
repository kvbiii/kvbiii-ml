import numpy as np
import pandas as pd
import pytest

from kvbiii_ml.data_processing.preprocessing.categorical_encoding.decision_tree_encoder import (
    DecisionTreeEncoderWithOriginal,
)


@pytest.fixture
def decision_tree_encoder_data() -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    """Provides categorical features with a regression and a classification target.

    Returns:
        tuple[pd.DataFrame, pd.Series, pd.Series]: Feature matrix, a continuous
            regression target, and a binary classification target.
    """
    X = pd.DataFrame(
        {
            "city": [
                "Warsaw",
                "Krakow",
                "Warsaw",
                "Gdansk",
                "Krakow",
                "Warsaw",
                "Gdansk",
                "Krakow",
            ],
            "tier": ["A", "B", "A", "C", "B", "A", "C", "B"],
        }
    )
    y_regression = pd.Series(
        [100.0, 200.0, 150.0, 80.0, 220.0, 130.0, 90.0, 210.0], name="target"
    )
    y_classification = pd.Series([1, 0, 1, 0, 1, 0, 0, 1], name="target")
    return X, y_regression, y_classification


def test_decisiontreeencoderwithoriginal_fit_returns_self(decision_tree_encoder_data):
    """Tests that fit() returns the encoder instance for scikit-learn compatibility.

    Args:
        decision_tree_encoder_data: Categorical features with regression/classification targets.

    Asserts:
        - fit() returns the same encoder instance it was called on.
    """
    X, y_regression, _ = decision_tree_encoder_data
    encoder = DecisionTreeEncoderWithOriginal(
        variables=["city", "tier"], regression=True, cv=2, random_state=17
    )
    result = encoder.fit(X, y_regression)
    if result is not encoder:
        raise AssertionError("fit() did not return self.")


def test_decisiontreeencoderwithoriginal_transform_regression_preserves_originals_and_appends(
    decision_tree_encoder_data,
):
    """Tests fit/transform with regression=True on a continuous target.

    Args:
        decision_tree_encoder_data: Categorical features with regression/classification targets.

    Asserts:
        - Original columns are preserved unchanged.
        - Derived columns are appended for every configured variable.
    """
    X, y_regression, _ = decision_tree_encoder_data
    encoder = DecisionTreeEncoderWithOriginal(
        variables=["city", "tier"], regression=True, cv=2, random_state=17
    )
    encoder.fit(X, y_regression)
    result = encoder.transform(X)
    pd.testing.assert_frame_equal(result[["city", "tier"]], X[["city", "tier"]])
    if "city_PREPROCESS_DT_ENC" not in result.columns:
        raise AssertionError("Missing derived column for city.")
    if "tier_PREPROCESS_DT_ENC" not in result.columns:
        raise AssertionError("Missing derived column for tier.")


def test_decisiontreeencoderwithoriginal_transform_classification_outputs_probabilities(
    decision_tree_encoder_data,
):
    """Tests fit/transform with regression=False on a binary classification target.

    Args:
        decision_tree_encoder_data: Categorical features with regression/classification targets.

    Asserts:
        - Derived column values fall within the valid probability range [0, 1].
    """
    X, _, y_classification = decision_tree_encoder_data
    encoder = DecisionTreeEncoderWithOriginal(
        variables=["city", "tier"], regression=False, cv=2, random_state=17
    )
    encoder.fit(X, y_classification)
    result = encoder.transform(X)
    values = result["city_PREPROCESS_DT_ENC"].to_numpy(dtype=float)
    if not np.all((values >= 0.0) & (values <= 1.0)):
        raise AssertionError("Classification-encoded values fell outside [0, 1].")


if __name__ == "__main__":
    print("Run this file with pytest to execute tests.")
