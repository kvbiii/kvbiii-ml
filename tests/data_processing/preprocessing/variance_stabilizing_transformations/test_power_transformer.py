import numpy as np
import pandas as pd
import pytest

from kvbiii_ml.data_processing.preprocessing.variance_stabilizing_transformations.power_transformer import (
    PowerTransformerWithOriginal,
)


@pytest.fixture
def positive_dataframe(test_settings) -> pd.DataFrame:
    """Provides a DataFrame with positive columns for power transformation tests.

    Args:
        test_settings: Test configuration fixture.

    Returns:
        pd.DataFrame: DataFrame with two positive columns.
    """
    rng = np.random.default_rng(test_settings.SEED)
    return pd.DataFrame(
        {
            "count": rng.integers(0, 500, test_settings.N_SAMPLES).astype(float),
            "duration_sec": rng.exponential(300, test_settings.N_SAMPLES),
        }
    )


def test_powertransformerwithoriginal_transform_defaults_to_square_root(
    positive_dataframe,
):
    """Tests that the default exp=0.5 produces the square-root transform.

    Args:
        positive_dataframe: Positive fixture DataFrame.

    Asserts:
        - Derived column values equal np.sqrt(x) within tolerance.
    """
    transformer = PowerTransformerWithOriginal(variables=["count", "duration_sec"])
    transformer.fit(positive_dataframe)
    result = transformer.transform(positive_dataframe)
    expected = np.sqrt(positive_dataframe["count"].to_numpy())
    if not np.allclose(result["count_PREPROCESS_POWER"].to_numpy(), expected):
        raise AssertionError("Default exp=0.5 does not match np.sqrt(x).")


def test_powertransformerwithoriginal_transform_matches_custom_exponent(
    positive_dataframe,
):
    """Tests that a custom exp value matches x ** exp.

    Args:
        positive_dataframe: Positive fixture DataFrame.

    Asserts:
        - Derived column values equal x ** exp within tolerance.
    """
    exponent = 2
    transformer = PowerTransformerWithOriginal(variables=["count"], exp=exponent)
    transformer.fit(positive_dataframe)
    result = transformer.transform(positive_dataframe)
    expected = positive_dataframe["count"].to_numpy() ** exponent
    if not np.allclose(result["count_PREPROCESS_POWER"].to_numpy(), expected):
        raise AssertionError("Custom exp value does not match x ** exp.")


def test_powertransformerwithoriginal_transform_preserves_original_columns(
    positive_dataframe,
):
    """Tests that transform keeps all original columns untouched.

    Args:
        positive_dataframe: Positive fixture DataFrame.

    Asserts:
        - Original columns are present in the output with identical values.
    """
    transformer = PowerTransformerWithOriginal(variables=["count", "duration_sec"])
    transformer.fit(positive_dataframe)
    result = transformer.transform(positive_dataframe)
    pd.testing.assert_frame_equal(
        result[positive_dataframe.columns.tolist()], positive_dataframe
    )


def test_powertransformerwithoriginal_transform_appends_derived_columns_named_correctly(
    positive_dataframe,
):
    """Tests that transform appends derived columns with the expected suffix.

    Args:
        positive_dataframe: Positive fixture DataFrame.

    Asserts:
        - Derived column names follow the ``{original}_PREPROCESS_POWER`` pattern.
    """
    transformer = PowerTransformerWithOriginal(variables=["count", "duration_sec"])
    transformer.fit(positive_dataframe)
    result = transformer.transform(positive_dataframe)
    if "count_PREPROCESS_POWER" not in result.columns:
        raise AssertionError("Expected derived column count_PREPROCESS_POWER missing.")
    if "duration_sec_PREPROCESS_POWER" not in result.columns:
        raise AssertionError(
            "Expected derived column duration_sec_PREPROCESS_POWER missing."
        )


if __name__ == "__main__":
    print("Run this file with pytest to execute tests.")
