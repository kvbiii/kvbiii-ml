import pandas as pd
import pytest

from kvbiii_ml.data_processing.preprocessing.discretisation.arbitrary_discretiser import (
    ArbitraryDiscretiserWithOriginal,
)


@pytest.fixture
def arbitrary_binning_data() -> pd.DataFrame:
    """Provides a DataFrame with an `age` column spanning known bin boundaries.

    Returns:
        pd.DataFrame: DataFrame with a single `age` numeric column.
    """
    return pd.DataFrame({"age": [5.0, 20.0, 30.0, 50.0]})


def test_arbitrarydiscretiserwithoriginal_transform_assigns_expected_bin_indices(
    arbitrary_binning_data,
):
    """Tests that transform assigns the expected bin index per explicit cut-point.

    Args:
        arbitrary_binning_data: DataFrame with an `age` column.

    Asserts:
        - Each value is assigned the bin index matching the configured cut-points.
        - The original `age` column is preserved unchanged.
    """
    binning_dict = {"age": [0, 10, 25, 40, 100]}
    discretiser = ArbitraryDiscretiserWithOriginal(binning_dict=binning_dict)
    discretiser.fit(arbitrary_binning_data)
    result = discretiser.transform(arbitrary_binning_data)
    expected_bins = pd.Series([0, 1, 2, 3], name="age_PREPROCESS_ARB_DISC")
    pd.testing.assert_series_equal(
        result["age_PREPROCESS_ARB_DISC"], expected_bins, check_dtype=False
    )
    pd.testing.assert_series_equal(result["age"], arbitrary_binning_data["age"])


def test_arbitrarydiscretiserwithoriginal_transform_errors_ignore_yields_nan_for_out_of_range(
    arbitrary_binning_data,
):
    """Tests that `errors="ignore"` produces NaN for values outside the cut-points.

    Args:
        arbitrary_binning_data: DataFrame with an `age` column.

    Asserts:
        - No exception is raised when an out-of-range value is transformed.
        - The derived column contains NaN for the out-of-range row.
        - The original column keeps the out-of-range value untouched.
    """
    binning_dict = {"age": [0, 10, 25, 40]}
    discretiser = ArbitraryDiscretiserWithOriginal(
        binning_dict=binning_dict, errors="ignore"
    )
    discretiser.fit(arbitrary_binning_data)
    out_of_range_data = pd.DataFrame({"age": [200.0]})
    result = discretiser.transform(out_of_range_data)
    if not pd.isna(result["age_PREPROCESS_ARB_DISC"].iloc[0]):
        raise AssertionError(
            "Out-of-range value should have produced NaN in derived column."
        )
    if result["age"].iloc[0] != 200.0:
        raise AssertionError("Original column value should remain untouched.")


def test_arbitrarydiscretiserwithoriginal_transform_errors_raise_raises_valueerror(
    arbitrary_binning_data,
):
    """Tests that `errors="raise"` raises a ValueError for out-of-range values.

    Args:
        arbitrary_binning_data: DataFrame with an `age` column.

    Asserts:
        - transform() raises ValueError when a value falls outside the cut-points.
    """
    binning_dict = {"age": [0, 10, 25, 40]}
    discretiser = ArbitraryDiscretiserWithOriginal(
        binning_dict=binning_dict, errors="raise"
    )
    discretiser.fit(arbitrary_binning_data)
    out_of_range_data = pd.DataFrame({"age": [200.0]})
    with pytest.raises(ValueError, match="NaN values were introduced"):
        discretiser.transform(out_of_range_data)


if __name__ == "__main__":
    print("Run this file with pytest to execute tests.")
