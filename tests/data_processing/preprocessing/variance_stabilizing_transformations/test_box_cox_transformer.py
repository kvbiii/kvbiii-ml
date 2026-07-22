import numpy as np
import pandas as pd
import pytest

from kvbiii_ml.data_processing.preprocessing.variance_stabilizing_transformations.box_cox_transformer import (
    BoxCoxTransformerWithOriginal,
)


@pytest.fixture
def positive_dataframe(test_settings) -> pd.DataFrame:
    """Provides a DataFrame with strictly positive columns for Box-Cox transformation tests.

    Args:
        test_settings: Test configuration fixture.

    Returns:
        pd.DataFrame: DataFrame with two strictly positive columns.
    """
    rng = np.random.default_rng(test_settings.SEED)
    return pd.DataFrame(
        {
            "income": rng.exponential(50_000, test_settings.N_SAMPLES) + 1,
            "price": rng.lognormal(3, 1, test_settings.N_SAMPLES) + 1,
        }
    )


@pytest.mark.parametrize("bad_value", [0, -1])
def test_boxcoxtransformerwithoriginal_fit_raises_valueerror_on_non_positive_values(
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
    transformer = BoxCoxTransformerWithOriginal(variables=["income"])
    with pytest.raises(ValueError, match="Data must be positive"):
        transformer.fit(non_positive)


def test_boxcoxtransformerwithoriginal_transform_preserves_original_columns(
    positive_dataframe,
):
    """Tests that transform keeps all original columns untouched.

    Args:
        positive_dataframe: Strictly positive fixture DataFrame.

    Asserts:
        - Original columns are present in the output with identical values.
    """
    transformer = BoxCoxTransformerWithOriginal(variables=["income", "price"])
    transformer.fit(positive_dataframe)
    result = transformer.transform(positive_dataframe)
    pd.testing.assert_frame_equal(
        result[positive_dataframe.columns.tolist()], positive_dataframe
    )


def test_boxcoxtransformerwithoriginal_transform_appends_derived_columns_named_correctly(
    positive_dataframe,
):
    """Tests that transform appends derived columns with the expected suffix.

    Args:
        positive_dataframe: Strictly positive fixture DataFrame.

    Asserts:
        - Derived column names follow the ``{original}_PREPROCESS_BOX_COX`` pattern.
    """
    transformer = BoxCoxTransformerWithOriginal(variables=["income", "price"])
    transformer.fit(positive_dataframe)
    result = transformer.transform(positive_dataframe)
    if "income_PREPROCESS_BOX_COX" not in result.columns:
        raise AssertionError(
            "Expected derived column income_PREPROCESS_BOX_COX missing."
        )
    if "price_PREPROCESS_BOX_COX" not in result.columns:
        raise AssertionError(
            "Expected derived column price_PREPROCESS_BOX_COX missing."
        )


def test_boxcoxtransformerwithoriginal_fit_stores_variables_used(positive_dataframe):
    """Tests that fit records which variables were used to fit the inner transformer.

    Args:
        positive_dataframe: Strictly positive fixture DataFrame.

    Asserts:
        - variables_ matches the requested variables.
    """
    transformer = BoxCoxTransformerWithOriginal(variables=["income", "price"])
    transformer.fit(positive_dataframe)
    if transformer.variables_ != ["income", "price"]:
        raise AssertionError("variables_ does not match the requested variables.")


if __name__ == "__main__":
    print("Run this file with pytest to execute tests.")
