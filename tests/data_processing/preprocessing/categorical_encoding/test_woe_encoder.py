import pandas as pd
import pytest

from kvbiii_ml.data_processing.preprocessing.categorical_encoding.woe_encoder import (
    WoEEncoderWithOriginal,
)


@pytest.fixture
def woe_balanced_data() -> tuple[pd.DataFrame, pd.Series]:
    """Provides a categorical column where every category has both target classes.

    Returns:
        tuple[pd.DataFrame, pd.Series]: Feature matrix with a `city` column and a
            binary target, where each city has at least one event and one non-event.
    """
    X = pd.DataFrame(
        {
            "city": [
                "Warsaw",
                "Warsaw",
                "Warsaw",
                "Warsaw",
                "Krakow",
                "Krakow",
                "Krakow",
                "Krakow",
            ]
        }
    )
    y = pd.Series([1, 1, 0, 0, 1, 0, 0, 0], name="target")
    return X, y


def test_woeencoderwithoriginal_fit_transform_binary_target(woe_balanced_data):
    """Tests normal fit/transform on a balanced binary target.

    Args:
        woe_balanced_data: DataFrame and binary target where every category has
            both classes represented.

    Asserts:
        - fit() returns the encoder instance.
        - Original columns are preserved unchanged.
        - The derived WoE column is appended.
    """
    X, y = woe_balanced_data
    encoder = WoEEncoderWithOriginal(variables=["city"])
    result_fit = encoder.fit(X, y)
    if result_fit is not encoder:
        raise AssertionError("fit() did not return self.")
    result = encoder.transform(X)
    pd.testing.assert_frame_equal(result[["city"]], X[["city"]])
    if "city_PREPROCESS_WOE" not in result.columns:
        raise AssertionError("Missing derived WoE column.")


def test_woeencoderwithoriginal_fit_raises_valueerror_when_category_missing_a_class():
    """Tests that fit() propagates a ValueError when a category lacks a target class.

    Weight of Evidence requires at least one event and one non-event per category.
    Warsaw here has only positive events and Krakow only negative ones.

    Asserts:
        - fit() raises ValueError due to the missing class per category.
    """
    X = pd.DataFrame(
        {"city": ["Warsaw", "Warsaw", "Warsaw", "Krakow", "Krakow", "Krakow"]}
    )
    y = pd.Series([1, 1, 1, 0, 0, 0], name="target")
    encoder = WoEEncoderWithOriginal(variables=["city"])
    with pytest.raises(ValueError):
        encoder.fit(X, y)


if __name__ == "__main__":
    print("Run this file with pytest to execute tests.")
