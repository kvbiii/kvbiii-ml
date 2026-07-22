import numpy as np
import pandas as pd
import pytest

from kvbiii_ml.data_processing.preprocessing.variance_stabilizing_transformations.arcsin_transformer import (
    ArcsinTransformerWithOriginal,
)


@pytest.fixture
def proportion_dataframe(test_settings) -> pd.DataFrame:
    """Provides a DataFrame with columns bounded in [0, 1] for arcsin transformation tests.

    Args:
        test_settings: Test configuration fixture.

    Returns:
        pd.DataFrame: DataFrame with two proportion-like columns in [0, 1].
    """
    rng = np.random.default_rng(test_settings.SEED)
    return pd.DataFrame(
        {
            "click_rate": rng.uniform(0.0, 1.0, test_settings.N_SAMPLES),
            "conversion_rate": rng.beta(2, 8, test_settings.N_SAMPLES),
        }
    )


def test_arcsintransformerwithoriginal_transform_preserves_original_columns(
    proportion_dataframe,
):
    """Tests that transform keeps all original columns untouched.

    Args:
        proportion_dataframe: Proportion-bounded fixture DataFrame.

    Asserts:
        - Original columns are present in the output with identical values.
    """
    transformer = ArcsinTransformerWithOriginal(
        variables=["click_rate", "conversion_rate"]
    )
    transformer.fit(proportion_dataframe)
    result = transformer.transform(proportion_dataframe)
    pd.testing.assert_frame_equal(
        result[proportion_dataframe.columns.tolist()], proportion_dataframe
    )


def test_arcsintransformerwithoriginal_transform_appends_derived_columns_named_correctly(
    proportion_dataframe,
):
    """Tests that transform appends derived columns with the expected suffix.

    Args:
        proportion_dataframe: Proportion-bounded fixture DataFrame.

    Asserts:
        - Derived column names follow the ``{original}_PREPROCESS_ARCSIN`` pattern.
    """
    transformer = ArcsinTransformerWithOriginal(
        variables=["click_rate", "conversion_rate"]
    )
    transformer.fit(proportion_dataframe)
    result = transformer.transform(proportion_dataframe)
    if "click_rate_PREPROCESS_ARCSIN" not in result.columns:
        raise AssertionError(
            "Expected derived column click_rate_PREPROCESS_ARCSIN missing."
        )
    if "conversion_rate_PREPROCESS_ARCSIN" not in result.columns:
        raise AssertionError(
            "Expected derived column conversion_rate_PREPROCESS_ARCSIN missing."
        )


def test_arcsintransformerwithoriginal_transform_matches_arcsin_sqrt_formula(
    proportion_dataframe,
):
    """Tests that derived values match arcsin(sqrt(x)).

    Args:
        proportion_dataframe: Proportion-bounded fixture DataFrame.

    Asserts:
        - Derived column values equal np.arcsin(np.sqrt(x)) within tolerance.
    """
    transformer = ArcsinTransformerWithOriginal(
        variables=["click_rate", "conversion_rate"]
    )
    transformer.fit(proportion_dataframe)
    result = transformer.transform(proportion_dataframe)
    expected = np.arcsin(np.sqrt(proportion_dataframe["click_rate"].to_numpy()))
    if not np.allclose(result["click_rate_PREPROCESS_ARCSIN"].to_numpy(), expected):
        raise AssertionError("Derived click_rate values do not match arcsin(sqrt(x)).")


def test_arcsintransformerwithoriginal_fit_raises_valueerror_outside_zero_one_range(
    proportion_dataframe,
):
    """Tests that fit raises when values fall outside the [0, 1] range.

    Args:
        proportion_dataframe: Proportion-bounded fixture DataFrame.

    Asserts:
        - ValueError is raised when a variable contains a value outside [0, 1].
    """
    out_of_range = proportion_dataframe.copy()
    out_of_range.loc[0, "click_rate"] = 1.5
    transformer = ArcsinTransformerWithOriginal(variables=["click_rate"])
    with pytest.raises(ValueError, match="outside the possible range"):
        transformer.fit(out_of_range)


if __name__ == "__main__":
    print("Run this file with pytest to execute tests.")
