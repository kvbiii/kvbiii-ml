import numpy as np
import pandas as pd
import pytest

from kvbiii_ml.data_processing.preprocessing.variance_stabilizing_transformations.log_cp_transformer import (
    LogCpTransformerWithOriginal,
)


@pytest.fixture
def skewed_with_zero_and_negative_dataframe(test_settings) -> pd.DataFrame:
    """Provides a DataFrame containing zero and negative values for LogCp transformation tests.

    Args:
        test_settings: Test configuration fixture.

    Returns:
        pd.DataFrame: DataFrame with a column containing zero, negative, and positive values.
    """
    rng = np.random.default_rng(test_settings.SEED)
    values = rng.exponential(10, test_settings.N_SAMPLES)
    values[0] = 0.0
    values[1] = -3.0
    return pd.DataFrame({"sales": values})


@pytest.mark.parametrize("base,numpy_log", [("e", np.log), ("10", np.log10)])
def test_logcptransformerwithoriginal_transform_matches_log_x_plus_c_formula(
    skewed_with_zero_and_negative_dataframe, base, numpy_log
):
    """Tests that derived values match log(x + C) for a fixed C and both supported bases.

    Args:
        skewed_with_zero_and_negative_dataframe: Fixture DataFrame with zero/negative values.
        base: Logarithm base passed to the transformer.
        numpy_log: NumPy logarithm function matching the base under test.

    Asserts:
        - Derived column values equal numpy_log(x + C) within tolerance.
    """
    constant_shift = 5.0
    transformer = LogCpTransformerWithOriginal(
        variables=["sales"], base=base, C=constant_shift
    )
    transformer.fit(skewed_with_zero_and_negative_dataframe)
    result = transformer.transform(skewed_with_zero_and_negative_dataframe)
    expected = numpy_log(
        skewed_with_zero_and_negative_dataframe["sales"].to_numpy() + constant_shift
    )
    if not np.allclose(result["sales_PREPROCESS_LOG_CP"].to_numpy(), expected):
        raise AssertionError("Derived sales values do not match log(x + C).")


def test_logcptransformerwithoriginal_fit_handles_auto_constant_on_zero_and_negative_values(
    skewed_with_zero_and_negative_dataframe,
):
    """Tests that C='auto' successfully fits a column containing zero and negative values.

    Args:
        skewed_with_zero_and_negative_dataframe: Fixture DataFrame with zero/negative values.

    Asserts:
        - No exception is raised during fit and transform.
        - The derived column contains no NaN or infinite values.
    """
    transformer = LogCpTransformerWithOriginal(variables=["sales"], C="auto")
    transformer.fit(skewed_with_zero_and_negative_dataframe)
    result = transformer.transform(skewed_with_zero_and_negative_dataframe)
    if not np.isfinite(result["sales_PREPROCESS_LOG_CP"].to_numpy()).all():
        raise AssertionError("Derived column with C='auto' contains non-finite values.")


def test_logcptransformerwithoriginal_transform_preserves_original_columns(
    skewed_with_zero_and_negative_dataframe,
):
    """Tests that transform keeps all original columns untouched.

    Args:
        skewed_with_zero_and_negative_dataframe: Fixture DataFrame with zero/negative values.

    Asserts:
        - Original columns are present in the output with identical values.
    """
    transformer = LogCpTransformerWithOriginal(variables=["sales"], C="auto")
    transformer.fit(skewed_with_zero_and_negative_dataframe)
    result = transformer.transform(skewed_with_zero_and_negative_dataframe)
    pd.testing.assert_frame_equal(
        result[skewed_with_zero_and_negative_dataframe.columns.tolist()],
        skewed_with_zero_and_negative_dataframe,
    )


if __name__ == "__main__":
    print("Run this file with pytest to execute tests.")
