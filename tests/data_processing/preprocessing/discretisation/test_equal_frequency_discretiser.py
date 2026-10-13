import numpy as np
import pandas as pd
import pytest

from kvbiii_ml.data_processing.preprocessing.discretisation.equal_frequency_discretiser import (
    EqualFrequencyDiscretiserWithOriginal,
)

N_ROWS = 40
CUSTOM_Q = 4


@pytest.fixture
def equal_frequency_data() -> pd.DataFrame:
    """Provides a uniformly distributed numeric DataFrame for quantile binning.

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


def test_equalfrequencydiscretiserwithoriginal_transform_preserves_original_columns(
    equal_frequency_data,
):
    """Tests that transform preserves the original columns exactly.

    Args:
        equal_frequency_data: Uniformly distributed numeric fixture.

    Asserts:
        - The original columns of the output match the input exactly.
    """
    discretiser = EqualFrequencyDiscretiserWithOriginal(variables=["age"], q=CUSTOM_Q)
    discretiser.fit(equal_frequency_data)
    result = discretiser.transform(equal_frequency_data)
    pd.testing.assert_frame_equal(
        result[equal_frequency_data.columns.tolist()], equal_frequency_data
    )


def test_equalfrequencydiscretiserwithoriginal_transform_produces_roughly_equal_bin_counts(
    equal_frequency_data,
):
    """Tests that a custom `q` produces roughly equal bin population counts.

    Args:
        equal_frequency_data: Uniformly distributed numeric fixture.

    Asserts:
        - Exactly `q` distinct bin codes are produced.
        - Every bin holds the same number of observations for evenly divisible data.
    """
    discretiser = EqualFrequencyDiscretiserWithOriginal(variables=["age"], q=CUSTOM_Q)
    discretiser.fit(equal_frequency_data)
    result = discretiser.transform(equal_frequency_data)
    bin_counts = result[f"age_PREPROCESS_EQ_FREQ_{CUSTOM_Q}"].value_counts()
    if len(bin_counts) != CUSTOM_Q:
        raise AssertionError(
            f"Expected {CUSTOM_Q} distinct bins, got {len(bin_counts)}."
        )
    expected_count_per_bin = N_ROWS // CUSTOM_Q
    if not (bin_counts == expected_count_per_bin).all():
        raise AssertionError(f"Bin counts not equal: {bin_counts.to_dict()}")


@pytest.mark.parametrize("q", [3, 4, 10])
def test_equalfrequencydiscretiserwithoriginal_suffix_encodes_q(
    equal_frequency_data, q
):
    """Tests that the derived column suffix is keyed on q, not a fixed string.

    Args:
        equal_frequency_data: Uniformly distributed numeric fixture.
        q: Bin count under test.

    Asserts:
        - The derived column is named "age_PREPROCESS_EQ_FREQ_{q}".
    """
    discretiser = EqualFrequencyDiscretiserWithOriginal(variables=["age"], q=q)
    discretiser.fit(equal_frequency_data)
    result = discretiser.transform(equal_frequency_data)
    expected_column = f"age_PREPROCESS_EQ_FREQ_{q}"
    if expected_column not in result.columns:
        raise AssertionError(f"Missing expected column {expected_column}.")


if __name__ == "__main__":
    print("Run this file with pytest to execute tests.")
