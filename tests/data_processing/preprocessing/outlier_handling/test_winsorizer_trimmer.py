import numpy as np
import pandas as pd

from kvbiii_ml.data_processing.preprocessing.outlier_handling.winsorizer_trimmer import (
    WinsorizerWithOriginal,
)


def test_winsorizerwithoriginal_default_params_caps_out_of_bound_values():
    """Tests default-params (iqr, both, fold=3.0) capping behavior.

    Asserts:
        - The derived column is capped (not NaN, not equal to the raw outlier
          value) at the outlier row, contrasting with OutlierTrimmer's NaN
          behavior.
        - Rows within bounds have derived values equal to their originals.
        - The original column and row count remain unchanged.
    """
    salary = pd.DataFrame({"salary": [50.0, 52.0, 51.0, 53.0, 200.0]})
    winsorizer = WinsorizerWithOriginal(variables=["salary"])
    winsorizer.fit(salary)
    result = winsorizer.transform(salary)
    derived = result["salary_PREPROCESS_WINSORIZER"]

    if len(result) != len(salary):
        raise AssertionError("Row count changed; Winsorizer must not drop rows.")
    if pd.isna(derived.iloc[-1]):
        raise AssertionError("Outlier row was NaN'd; Winsorizer should cap, not NaN.")
    if derived.iloc[-1] == salary["salary"].iloc[-1]:
        raise AssertionError("Outlier row was not capped.")
    if derived.iloc[-1] >= salary["salary"].iloc[-1]:
        raise AssertionError("Capped value should be lower than the raw outlier.")
    for position in range(len(salary) - 1):
        if derived.iloc[position] != salary["salary"].iloc[position]:
            raise AssertionError(f"In-bound row {position} should be unchanged.")
    pd.testing.assert_series_equal(result["salary"], salary["salary"])


def test_winsorizerwithoriginal_transform_ignores_missing_values():
    """Tests that missing_values='ignore' is honored and NaNs don't raise.

    Asserts:
        - transform() does not raise when the target variable contains NaN.
        - The NaN value is preserved (not imputed) in the derived column.
        - Non-missing rows are still processed normally.
    """
    salary = pd.DataFrame({"salary": [50.0, 52.0, np.nan, 53.0, 200.0]})
    winsorizer = WinsorizerWithOriginal(variables=["salary"], fold=1.0)
    winsorizer.fit(salary)
    result = winsorizer.transform(salary)
    derived = result["salary_PREPROCESS_WINSORIZER"]

    if not pd.isna(derived.iloc[2]):
        raise AssertionError("Missing value should remain NaN in the derived column.")
    if pd.isna(derived.iloc[-1]):
        raise AssertionError("Non-missing outlier row should still be capped, not NaN.")


def test_winsorizerwithoriginal_transform_preserves_original_columns():
    """Tests that the original column is preserved untouched after capping.

    Asserts:
        - The original salary column is byte-for-byte unchanged after transform,
          even though the derived column reflects capped values.
    """
    salary = pd.DataFrame({"salary": [10.0, 12.0, 11.0, 13.0, 12.0, 100.0]})
    winsorizer = WinsorizerWithOriginal(
        variables=["salary"], capping_method="iqr", fold=1.0
    )
    winsorizer.fit(salary)
    result = winsorizer.transform(salary)
    pd.testing.assert_series_equal(result["salary"], salary["salary"])
    if "salary_PREPROCESS_WINSORIZER" not in result.columns:
        raise AssertionError("Derived column missing from transform output.")


if __name__ == "__main__":
    print("Run this file with pytest to execute tests.")
