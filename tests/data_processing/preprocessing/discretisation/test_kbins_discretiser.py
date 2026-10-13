import warnings

import numpy as np
import pandas as pd
import pytest

from kvbiii_ml.data_processing.preprocessing.discretisation.kbins_discretiser import (
    KBinsDiscretiserWithOriginal,
)


@pytest.fixture
def mixed_dtype_data() -> pd.DataFrame:
    """Provides a DataFrame mixing numeric and non-numeric columns.

    Returns:
        pd.DataFrame: One evenly-spaced numeric column and one string column.
    """
    return pd.DataFrame(
        {
            "score": np.arange(20, dtype=float),
            "city": ["Warsaw", "Krakow"] * 10,
        }
    )


@pytest.fixture
def data_with_nan() -> pd.DataFrame:
    """Provides a DataFrame with a numeric column containing NaN values.

    Returns:
        pd.DataFrame: A "score" column with 12 non-null values and 3 NaNs
            interspersed at the end.
    """
    values = list(range(12)) + [np.nan, np.nan, np.nan]
    return pd.DataFrame({"score": values})


@pytest.mark.parametrize("n_bins", [2, 5, 10])
def test_kbinsdiscretiserwithoriginal_suffix_encodes_n_bins(mixed_dtype_data, n_bins):
    """Tests that the derived column suffix is keyed on n_bins, not a fixed string.

    Args:
        mixed_dtype_data: Fixture with a numeric "score" column.
        n_bins: Bin count under test.

    Asserts:
        - The derived column is named "score_PREPROCESS_KBINS_{n_bins}".
    """
    discretiser = KBinsDiscretiserWithOriginal(n_bins=n_bins, variables=["score"])
    discretiser.fit(mixed_dtype_data)
    result = discretiser.transform(mixed_dtype_data)
    expected_column = f"score_PREPROCESS_KBINS_{n_bins}"
    if expected_column not in result.columns:
        raise AssertionError(f"Missing expected column {expected_column}.")


def test_kbinsdiscretiserwithoriginal_defaults_to_numeric_columns_only(
    mixed_dtype_data,
):
    """Tests that auto-detection (variables=None) selects only numeric columns.

    Args:
        mixed_dtype_data: Fixture mixing a numeric and a string column.

    Asserts:
        - variables_ contains "score" but not the non-numeric "city" column.
    """
    discretiser = KBinsDiscretiserWithOriginal(n_bins=4)
    discretiser.fit(mixed_dtype_data)
    if discretiser.variables_ != ["score"]:
        raise AssertionError(f"Expected ['score'], got {discretiser.variables_}.")


def test_kbinsdiscretiserwithoriginal_quantile_strategy_produces_balanced_bins(
    mixed_dtype_data,
):
    """Tests that quantile binning splits evenly-spaced data into equal-sized bins.

    Args:
        mixed_dtype_data: Fixture with 20 evenly-spaced "score" values.

    Asserts:
        - Each of the 4 quantile bins receives exactly 5 rows.
    """
    discretiser = KBinsDiscretiserWithOriginal(
        n_bins=4, strategy="quantile", variables=["score"]
    )
    discretiser.fit(mixed_dtype_data)
    result = discretiser.transform(mixed_dtype_data)
    bin_counts = result["score_PREPROCESS_KBINS_4"].value_counts()
    if not (bin_counts == 5).all():
        raise AssertionError(
            f"Expected 4 equal-sized bins of 5, got {bin_counts.to_dict()}."
        )


def test_kbinsdiscretiserwithoriginal_raises_on_nan_by_default(data_with_nan):
    """Tests that fit raises when NaNs are present and missing_values="raise".

    Args:
        data_with_nan: Fixture with a "score" column containing NaN values.

    Asserts:
        - fit() raises ValueError, matching KBinsDiscretizer's own NaN error.
    """
    discretiser = KBinsDiscretiserWithOriginal(n_bins=3, variables=["score"])
    with pytest.raises(ValueError, match="NaN"):
        discretiser.fit(data_with_nan)


def test_kbinsdiscretiserwithoriginal_ignore_missing_values_preserves_nan(
    data_with_nan,
):
    """Tests that missing_values="ignore" leaves NaNs as NaN and bins the rest.

    Args:
        data_with_nan: Fixture with a "score" column containing NaN values.

    Asserts:
        - Rows that were NaN in "score" are still NaN in the derived column.
        - Rows that were not NaN are assigned a finite bin label.
    """
    discretiser = KBinsDiscretiserWithOriginal(
        n_bins=3, variables=["score"], missing_values="ignore"
    )
    discretiser.fit(data_with_nan)
    result = discretiser.transform(data_with_nan)
    derived = result["score_PREPROCESS_KBINS_3"]
    nan_mask = data_with_nan["score"].isna()
    if not derived[nan_mask].isna().all():
        raise AssertionError(
            "Expected NaN rows in source to remain NaN in derived column."
        )
    if derived[~nan_mask].isna().any():
        raise AssertionError("Expected non-NaN rows in source to be binned, not NaN.")


def test_kbinsdiscretiserwithoriginal_ignore_missing_values_all_nan_column_fits():
    """Tests that an entirely-NaN column still fits when missing_values="ignore".

    Asserts:
        - fit_transform() does not raise on an all-NaN column.
        - The all-NaN column's derived column is entirely NaN.
        - A sibling column with data is still binned normally.
    """
    all_nan_data = pd.DataFrame(
        {"score": [np.nan, np.nan, np.nan, np.nan], "age": [1.0, 2.0, 3.0, 4.0]}
    )
    discretiser = KBinsDiscretiserWithOriginal(
        n_bins=2, variables=["score", "age"], missing_values="ignore"
    )
    result = discretiser.fit_transform(all_nan_data)
    if not result["score_PREPROCESS_KBINS_2"].isna().all():
        raise AssertionError(
            "Expected all-NaN source column to yield an all-NaN derived column."
        )
    if result["age_PREPROCESS_KBINS_2"].isna().any():
        raise AssertionError(
            "Expected the non-null sibling column to be binned, not NaN."
        )


def test_kbinsdiscretiserwithoriginal_all_nan_at_fit_stays_nan_on_new_data():
    """Tests that a column all-NaN at fit time stays NaN even if transform sees values.

    Asserts:
        - Transforming data where the column now has values yields NaN, since no
          bin edges were learned for it.
    """
    discretiser = KBinsDiscretiserWithOriginal(
        n_bins=2, variables=["score"], missing_values="ignore"
    )
    discretiser.fit(pd.DataFrame({"score": [np.nan, np.nan, np.nan]}))
    result = discretiser.transform(pd.DataFrame({"score": [1.0, 2.0, np.nan]}))
    if not result["score_PREPROCESS_KBINS_2"].isna().all():
        raise AssertionError("Expected a column with no learned bin edges to stay NaN.")


def test_kbinsdiscretiserwithoriginal_constant_column_emits_no_warning():
    """Tests that binning a constant column stays silent instead of warning.

    KBinsDiscretizer falls back to a single bin for a constant column on its
    own - the UserWarning it emits about this is expected, benign fallback
    behavior, not something callers of this wrapper need surfaced.

    Asserts:
        - fit_transform() on an all-identical-value column raises no warnings.
    """
    constant_data = pd.DataFrame({"score": [5.0] * 10})
    discretiser = KBinsDiscretiserWithOriginal(n_bins=3, variables=["score"])
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        discretiser.fit_transform(constant_data)
    if caught:
        raise AssertionError(
            f"Expected no warnings, got: {[str(w.message) for w in caught]}"
        )


if __name__ == "__main__":
    print("Run this file with pytest to execute tests.")
