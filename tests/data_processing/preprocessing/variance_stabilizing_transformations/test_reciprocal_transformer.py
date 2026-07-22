import numpy as np
import pandas as pd
import pytest

from kvbiii_ml.data_processing.preprocessing.variance_stabilizing_transformations.reciprocal_transformer import (
    ReciprocalTransformerWithOriginal,
)


@pytest.fixture
def nonzero_dataframe(test_settings) -> pd.DataFrame:
    """Provides a DataFrame with non-zero columns for reciprocal transformation tests.

    Args:
        test_settings: Test configuration fixture.

    Returns:
        pd.DataFrame: DataFrame with two non-zero columns.
    """
    rng = np.random.default_rng(test_settings.SEED)
    return pd.DataFrame(
        {
            "response_time_ms": rng.exponential(200, test_settings.N_SAMPLES) + 1,
            "request_size_kb": rng.exponential(50, test_settings.N_SAMPLES) + 1,
        }
    )


def test_reciprocaltransformerwithoriginal_transform_matches_reciprocal_formula(
    nonzero_dataframe,
):
    """Tests that derived values match 1/x.

    Args:
        nonzero_dataframe: Non-zero fixture DataFrame.

    Asserts:
        - Derived column values equal 1 / x within tolerance.
    """
    transformer = ReciprocalTransformerWithOriginal(
        variables=["response_time_ms", "request_size_kb"]
    )
    transformer.fit(nonzero_dataframe)
    result = transformer.transform(nonzero_dataframe)
    expected = 1.0 / nonzero_dataframe["response_time_ms"].to_numpy()
    if not np.allclose(
        result["response_time_ms_PREPROCESS_RECIPROCAL"].to_numpy(), expected
    ):
        raise AssertionError("Derived response_time_ms values do not match 1/x.")


def test_reciprocaltransformerwithoriginal_transform_preserves_original_columns(
    nonzero_dataframe,
):
    """Tests that transform keeps all original columns untouched.

    Args:
        nonzero_dataframe: Non-zero fixture DataFrame.

    Asserts:
        - Original columns are present in the output with identical values.
    """
    transformer = ReciprocalTransformerWithOriginal(
        variables=["response_time_ms", "request_size_kb"]
    )
    transformer.fit(nonzero_dataframe)
    result = transformer.transform(nonzero_dataframe)
    pd.testing.assert_frame_equal(
        result[nonzero_dataframe.columns.tolist()], nonzero_dataframe
    )


def test_reciprocaltransformerwithoriginal_transform_appends_derived_columns_named_correctly(
    nonzero_dataframe,
):
    """Tests that transform appends derived columns with the expected suffix.

    Args:
        nonzero_dataframe: Non-zero fixture DataFrame.

    Asserts:
        - Derived column names follow the ``{original}_PREPROCESS_RECIPROCAL`` pattern.
    """
    transformer = ReciprocalTransformerWithOriginal(
        variables=["response_time_ms", "request_size_kb"]
    )
    transformer.fit(nonzero_dataframe)
    result = transformer.transform(nonzero_dataframe)
    if "response_time_ms_PREPROCESS_RECIPROCAL" not in result.columns:
        raise AssertionError(
            "Expected derived column response_time_ms_PREPROCESS_RECIPROCAL missing."
        )
    if "request_size_kb_PREPROCESS_RECIPROCAL" not in result.columns:
        raise AssertionError(
            "Expected derived column request_size_kb_PREPROCESS_RECIPROCAL missing."
        )


def test_reciprocaltransformerwithoriginal_fit_raises_valueerror_on_zero_value(
    nonzero_dataframe,
):
    """Tests that fit raises when a variable contains the value zero.

    Args:
        nonzero_dataframe: Non-zero fixture DataFrame.

    Asserts:
        - ValueError is raised when a variable contains zero.
    """
    with_zero = nonzero_dataframe.copy()
    with_zero.loc[0, "response_time_ms"] = 0.0
    transformer = ReciprocalTransformerWithOriginal(variables=["response_time_ms"])
    with pytest.raises(ValueError, match="value zero"):
        transformer.fit(with_zero)


if __name__ == "__main__":
    print("Run this file with pytest to execute tests.")
