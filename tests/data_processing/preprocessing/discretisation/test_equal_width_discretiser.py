import numpy as np
import pandas as pd
import pytest

from kvbiii_ml.data_processing.preprocessing.discretisation.equal_width_discretiser import (
    EqualWidthDiscretiserWithOriginal,
)

N_ROWS = 40
CUSTOM_BINS = 4


@pytest.fixture
def equal_width_data() -> pd.DataFrame:
    """Provides a uniformly distributed numeric DataFrame for width-based binning.

    Returns:
        pd.DataFrame: DataFrame with `age` and `income` numeric columns.
    """
    rng = np.random.default_rng(42)
    return pd.DataFrame(
        {
            "age": rng.uniform(0, 100, N_ROWS),
            "income": rng.uniform(0, 100, N_ROWS),
        }
    )


def test_equalwidthdiscretiserwithoriginal_transform_preserves_original_columns(
    equal_width_data,
):
    """Tests that transform preserves the original columns exactly.

    Args:
        equal_width_data: Uniformly distributed numeric fixture.

    Asserts:
        - The original columns of the output match the input exactly.
    """
    discretiser = EqualWidthDiscretiserWithOriginal(variables=["age"], bins=CUSTOM_BINS)
    discretiser.fit(equal_width_data)
    result = discretiser.transform(equal_width_data)
    pd.testing.assert_frame_equal(result[equal_width_data.columns.tolist()], equal_width_data)


def test_equalwidthdiscretiserwithoriginal_transform_produces_expected_distinct_bin_count(
    equal_width_data,
):
    """Tests that a custom `bins` value yields the expected number of distinct bin codes.

    Args:
        equal_width_data: Uniformly distributed numeric fixture.

    Asserts:
        - Exactly `bins` distinct bin codes appear in the derived column.
        - Bin codes are contiguous starting at 0.
    """
    discretiser = EqualWidthDiscretiserWithOriginal(variables=["age"], bins=CUSTOM_BINS)
    discretiser.fit(equal_width_data)
    result = discretiser.transform(equal_width_data)
    distinct_bins = sorted(result["age_PREPROCESS_EQ_WIDTH"].unique())
    if distinct_bins != list(range(CUSTOM_BINS)):
        raise AssertionError(f"Expected bin codes {list(range(CUSTOM_BINS))}, got {distinct_bins}.")


if __name__ == "__main__":
    print("Run this file with pytest to execute tests.")
