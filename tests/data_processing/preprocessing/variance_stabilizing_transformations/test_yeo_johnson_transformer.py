import numpy as np
import pandas as pd
import pytest

from kvbiii_ml.data_processing.preprocessing.variance_stabilizing_transformations.yeo_johnson_transformer import (
    YeoJohnsonTransformerWithOriginal,
)


@pytest.fixture
def zero_and_negative_dataframe(test_settings) -> pd.DataFrame:
    """Provides a DataFrame containing zero and negative values for Yeo-Johnson tests.

    Args:
        test_settings: Test configuration fixture.

    Returns:
        pd.DataFrame: DataFrame with a column spanning negative, zero, and positive values.
    """
    rng = np.random.default_rng(test_settings.SEED)
    return pd.DataFrame(
        {
            "score": rng.normal(0, 5, test_settings.N_SAMPLES),
            "balance": rng.normal(1_000, 5_000, test_settings.N_SAMPLES),
        }
    )


def test_yeojohnsontransformerwithoriginal_fit_transform_succeeds_on_zero_and_negative_values(
    zero_and_negative_dataframe,
):
    """Tests that Yeo-Johnson, unlike Box-Cox, handles zero and negative values without error.

    Args:
        zero_and_negative_dataframe: Fixture DataFrame spanning negative, zero, and positive
            values.

    Asserts:
        - The fixture indeed contains a negative value, proving the scenario is meaningful.
        - No exception is raised during fit or transform.
        - The derived column contains no NaN or infinite values.
    """
    contains_negative = (zero_and_negative_dataframe["score"] < 0).any()
    if not contains_negative:
        raise AssertionError("Fixture must contain at least one negative value.")
    zero_and_negative_dataframe.loc[0, "score"] = 0.0
    transformer = YeoJohnsonTransformerWithOriginal(variables=["score", "balance"])
    transformer.fit(zero_and_negative_dataframe)
    result = transformer.transform(zero_and_negative_dataframe)
    if not np.isfinite(result["score_PREPROCESS_YEO_JOHNSON"].to_numpy()).all():
        raise AssertionError("Derived score column contains non-finite values.")
    if not np.isfinite(result["balance_PREPROCESS_YEO_JOHNSON"].to_numpy()).all():
        raise AssertionError("Derived balance column contains non-finite values.")


def test_yeojohnsontransformerwithoriginal_transform_preserves_original_columns(
    zero_and_negative_dataframe,
):
    """Tests that transform keeps all original columns untouched.

    Args:
        zero_and_negative_dataframe: Fixture DataFrame spanning negative, zero, and positive
            values.

    Asserts:
        - Original columns are present in the output with identical values.
    """
    transformer = YeoJohnsonTransformerWithOriginal(variables=["score", "balance"])
    transformer.fit(zero_and_negative_dataframe)
    result = transformer.transform(zero_and_negative_dataframe)
    pd.testing.assert_frame_equal(
        result[zero_and_negative_dataframe.columns.tolist()],
        zero_and_negative_dataframe,
    )


def test_yeojohnsontransformerwithoriginal_transform_appends_derived_columns_named_correctly(
    zero_and_negative_dataframe,
):
    """Tests that transform appends derived columns with the expected suffix.

    Args:
        zero_and_negative_dataframe: Fixture DataFrame spanning negative, zero, and positive
            values.

    Asserts:
        - Derived column names follow the ``{original}_PREPROCESS_YEO_JOHNSON`` pattern.
    """
    transformer = YeoJohnsonTransformerWithOriginal(variables=["score", "balance"])
    transformer.fit(zero_and_negative_dataframe)
    result = transformer.transform(zero_and_negative_dataframe)
    if "score_PREPROCESS_YEO_JOHNSON" not in result.columns:
        raise AssertionError(
            "Expected derived column score_PREPROCESS_YEO_JOHNSON missing."
        )
    if "balance_PREPROCESS_YEO_JOHNSON" not in result.columns:
        raise AssertionError(
            "Expected derived column balance_PREPROCESS_YEO_JOHNSON missing."
        )


if __name__ == "__main__":
    print("Run this file with pytest to execute tests.")
