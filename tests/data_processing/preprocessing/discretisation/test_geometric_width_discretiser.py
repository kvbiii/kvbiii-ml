import numpy as np
import pandas as pd
import pytest

from kvbiii_ml.data_processing.preprocessing.discretisation.geometric_width_discretiser import (
    GeometricWidthDiscretiserWithOriginal,
)

N_ROWS = 40
CUSTOM_BINS = 4


@pytest.fixture
def positive_income_data() -> pd.DataFrame:
    """Provides a strictly positive, right-skewed numeric DataFrame.

    Returns:
        pd.DataFrame: DataFrame with a single strictly positive `income` column.
    """
    rng = np.random.default_rng(42)
    return pd.DataFrame({"income": rng.exponential(scale=50.0, size=N_ROWS) + 1.0})


@pytest.fixture
def non_positive_data() -> pd.DataFrame:
    """Provides a constant, non-positive numeric DataFrame.

    Returns:
        pd.DataFrame: DataFrame with a single constant `income` column of zeros.
    """
    return pd.DataFrame({"income": [0.0, 0.0, 0.0, 0.0]})


def test_geometricwidthdiscretiserwithoriginal_transform_preserves_original_columns(
    positive_income_data,
):
    """Tests that transform preserves the original columns exactly on positive data.

    Args:
        positive_income_data: Strictly positive numeric fixture.

    Asserts:
        - The original columns of the output match the input exactly.
    """
    discretiser = GeometricWidthDiscretiserWithOriginal(
        variables=["income"], bins=CUSTOM_BINS
    )
    discretiser.fit(positive_income_data)
    result = discretiser.transform(positive_income_data)
    pd.testing.assert_frame_equal(
        result[positive_income_data.columns.tolist()], positive_income_data
    )


def test_geometricwidthdiscretiserwithoriginal_transform_produces_expected_distinct_bin_count(
    positive_income_data,
):
    """Tests that a custom `bins` value yields the expected number of distinct bin codes.

    Args:
        positive_income_data: Strictly positive numeric fixture.

    Asserts:
        - Exactly `bins` distinct bin codes appear in the derived column.
    """
    discretiser = GeometricWidthDiscretiserWithOriginal(
        variables=["income"], bins=CUSTOM_BINS
    )
    discretiser.fit(positive_income_data)
    result = discretiser.transform(positive_income_data)
    distinct_bins = sorted(result["income_PREPROCESS_GEO_WIDTH"].unique())
    if distinct_bins != list(range(CUSTOM_BINS)):
        raise AssertionError(
            f"Expected bin codes {list(range(CUSTOM_BINS))}, got {distinct_bins}."
        )


def test_geometricwidthdiscretiserwithoriginal_transform_raises_for_non_positive_data(
    non_positive_data,
):
    """Tests that non-positive (constant zero) data raises a ValueError at transform time.

    A constant zero column collapses the geometric progression into duplicate bin
    edges, which feature_engine's underlying `pd.cut` call rejects.

    Args:
        non_positive_data: Constant, non-positive numeric fixture.

    Asserts:
        - transform() raises ValueError for constant non-positive input.
    """
    discretiser = GeometricWidthDiscretiserWithOriginal(variables=["income"], bins=3)
    discretiser.fit(non_positive_data)
    with pytest.raises(ValueError, match="Bin edges must be unique"):
        discretiser.transform(non_positive_data)


if __name__ == "__main__":
    print("Run this file with pytest to execute tests.")
