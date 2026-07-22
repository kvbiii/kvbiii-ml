import numpy as np
import pandas as pd
import pytest

from kvbiii_ml.data_processing.preprocessing.variance_stabilizing_transformations.log_transformer import (
    LogTransformerWithOriginal,
)


@pytest.fixture
def positive_dataframe(test_settings) -> pd.DataFrame:
    """Provides a DataFrame with strictly positive columns for log transformation tests.

    Args:
        test_settings: Test configuration fixture.

    Returns:
        pd.DataFrame: DataFrame with two strictly positive columns.
    """
    rng = np.random.default_rng(test_settings.SEED)
    return pd.DataFrame(
        {
            "income": rng.exponential(50_000, test_settings.N_SAMPLES) + 1,
            "tenure_days": rng.exponential(200, test_settings.N_SAMPLES) + 1,
        }
    )


@pytest.mark.parametrize("base,numpy_log", [("e", np.log), ("10", np.log10)])
def test_logtransformerwithoriginal_transform_matches_log_formula(
    positive_dataframe, base, numpy_log
):
    """Tests that derived values match the log transform for both supported bases.

    Args:
        positive_dataframe: Strictly positive fixture DataFrame.
        base: Logarithm base passed to the transformer.
        numpy_log: NumPy logarithm function matching the base under test.

    Asserts:
        - Derived column values equal numpy_log(x) within tolerance.
    """
    transformer = LogTransformerWithOriginal(variables=["income"], base=base)
    transformer.fit(positive_dataframe)
    result = transformer.transform(positive_dataframe)
    expected = numpy_log(positive_dataframe["income"].to_numpy())
    if not np.allclose(result["income_PREPROCESS_LOG"].to_numpy(), expected):
        raise AssertionError("Derived income values do not match the log formula.")


def test_logtransformerwithoriginal_transform_preserves_original_columns(
    positive_dataframe,
):
    """Tests that transform keeps all original columns untouched.

    Args:
        positive_dataframe: Strictly positive fixture DataFrame.

    Asserts:
        - Original columns are present in the output with identical values.
    """
    transformer = LogTransformerWithOriginal(variables=["income", "tenure_days"])
    transformer.fit(positive_dataframe)
    result = transformer.transform(positive_dataframe)
    pd.testing.assert_frame_equal(
        result[positive_dataframe.columns.tolist()], positive_dataframe
    )


def test_logtransformerwithoriginal_transform_appends_derived_columns_named_correctly(
    positive_dataframe,
):
    """Tests that transform appends derived columns with the expected suffix.

    Args:
        positive_dataframe: Strictly positive fixture DataFrame.

    Asserts:
        - Derived column names follow the ``{original}_PREPROCESS_LOG`` pattern.
    """
    transformer = LogTransformerWithOriginal(variables=["income", "tenure_days"])
    transformer.fit(positive_dataframe)
    result = transformer.transform(positive_dataframe)
    if "income_PREPROCESS_LOG" not in result.columns:
        raise AssertionError("Expected derived column income_PREPROCESS_LOG missing.")
    if "tenure_days_PREPROCESS_LOG" not in result.columns:
        raise AssertionError(
            "Expected derived column tenure_days_PREPROCESS_LOG missing."
        )


@pytest.mark.parametrize("bad_value", [0, -1])
def test_logtransformerwithoriginal_fit_raises_valueerror_on_non_positive_values(
    positive_dataframe, bad_value
):
    """Tests that fit raises when a variable contains a zero or negative value.

    Args:
        positive_dataframe: Strictly positive fixture DataFrame.
        bad_value: Non-positive value injected into the fixture.

    Asserts:
        - ValueError is raised for zero and negative inputs.
    """
    non_positive = positive_dataframe.copy()
    non_positive.loc[0, "income"] = bad_value
    transformer = LogTransformerWithOriginal(variables=["income"])
    with pytest.raises(ValueError, match="zero or negative values"):
        transformer.fit(non_positive)


if __name__ == "__main__":
    print("Run this file with pytest to execute tests.")
