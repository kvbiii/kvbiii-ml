import pandas as pd
import pytest

from kvbiii_ml.data_processing.preprocessing.outlier_handling.outlier_trimmer import (
    OutlierTrimmerWithOriginal,
)


@pytest.fixture
def salary_with_outlier() -> pd.DataFrame:
    """Provides a single-column DataFrame with a clear right-tail outlier.

    Returns:
        pd.DataFrame: DataFrame with a `salary` column where the last row is a
            clear outlier relative to the other four values.
    """
    return pd.DataFrame({"salary": [50.0, 52.0, 51.0, 53.0, 200.0]})


@pytest.fixture
def salary_with_both_tail_outliers() -> pd.DataFrame:
    """Provides a single-column DataFrame with both a low and a high outlier.

    Returns:
        pd.DataFrame: DataFrame with a `salary` column where the first and last
            rows are clear outliers relative to the middle three values.
    """
    return pd.DataFrame({"salary": [-500.0, 50.0, 52.0, 51.0, 200.0]})


@pytest.mark.parametrize("capping_method", ["gaussian", "iqr", "mad", "quantiles"])
def test_outliertrimmerwithoriginal_transform_nans_out_of_bound_rows_across_capping_methods(
    salary_with_outlier, capping_method
):
    """Tests that out-of-bound values are NaN'd in the derived column across capping methods.

    Args:
        salary_with_outlier: DataFrame with a clear right-tail outlier.
        capping_method: Configured `capping_method` value under test.

    Asserts:
        - The derived column is NaN at the outlier row for every capping method.
        - The original `salary` column and row count remain unchanged.
    """
    kwargs = {"capping_method": capping_method, "tail": "right"}
    if capping_method == "quantiles":
        kwargs["fold"] = 0.1
    if capping_method == "gaussian":
        kwargs["fold"] = 1.0
    trimmer = OutlierTrimmerWithOriginal(variables=["salary"], **kwargs)
    trimmer.fit(salary_with_outlier)
    result = trimmer.transform(salary_with_outlier)
    if len(result) != len(salary_with_outlier):
        raise AssertionError("Row count changed; no rows should be dropped.")
    if not pd.isna(result["salary_PREPROCESS_TRIMMED"].iloc[-1]):
        raise AssertionError(
            f"Outlier row was not NaN'd for capping_method={capping_method}."
        )
    pd.testing.assert_series_equal(result["salary"], salary_with_outlier["salary"])


@pytest.mark.parametrize(
    "tail, outlier_positions",
    [("right", [4]), ("left", [0]), ("both", [0, 4])],
)
def test_outliertrimmerwithoriginal_transform_nans_expected_tail_across_tail_variants(
    salary_with_both_tail_outliers, tail, outlier_positions
):
    """Tests that only the configured tail's outliers are NaN'd in the derived column.

    Args:
        salary_with_both_tail_outliers: DataFrame with a low and a high outlier.
        tail: Configured `tail` value under test.
        outlier_positions: Row positions expected to be NaN'd for this tail.

    Asserts:
        - Every row count is preserved regardless of tail configuration.
        - Only rows within the configured tail are NaN'd in the derived column.
        - The original `salary` column remains unchanged.
    """
    trimmer = OutlierTrimmerWithOriginal(
        variables=["salary"], capping_method="iqr", tail=tail, fold=1.0
    )
    trimmer.fit(salary_with_both_tail_outliers)
    result = trimmer.transform(salary_with_both_tail_outliers)
    if len(result) != len(salary_with_both_tail_outliers):
        raise AssertionError("Row count changed; no rows should be dropped.")
    derived = result["salary_PREPROCESS_TRIMMED"]
    for position in range(len(salary_with_both_tail_outliers)):
        if position in outlier_positions:
            if not pd.isna(derived.iloc[position]):
                raise AssertionError(f"Row {position} should be NaN for tail={tail}.")
        else:
            if pd.isna(derived.iloc[position]):
                raise AssertionError(
                    f"Row {position} should not be NaN for tail={tail}."
                )
    pd.testing.assert_series_equal(
        result["salary"], salary_with_both_tail_outliers["salary"]
    )


if __name__ == "__main__":
    print("Run this file with pytest to execute tests.")
