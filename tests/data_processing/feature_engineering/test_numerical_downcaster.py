"""Tests for kvbiii_ml.data_processing.feature_engineering.numerical_downcaster module."""

import numpy as np
import pandas as pd
import pytest
from sklearn.exceptions import NotFittedError

from kvbiii_ml.data_processing.feature_engineering.numerical_downcaster import (
    NumericalDowncaster,
)


def test_numericaldowncaster_fit_downcasts_nan_containing_integer_column_to_float32():
    """Tests fit assigns float32 to a nullable-integer column that contains NaN values.

    Asserts:
        - The learned target dtype for the NaN-containing column is np.float32
    """
    df = pd.DataFrame({"a": pd.array([1, 2, pd.NA, 4], dtype="Int64")})
    downcaster = NumericalDowncaster()

    downcaster.fit(df)

    if downcaster.dtype_map_["a"] != np.float32:
        raise AssertionError()


@pytest.mark.parametrize(
    "values,expected_dtype",
    [
        ([-128, 0, 127], np.int8),
        ([-129, 0, 127], np.int16),
        ([-128, 0, 128], np.int16),
        ([-32768, 0, 32767], np.int16),
        ([-32769, 0, 32767], np.int32),
        ([-2147483648, 0, 2147483647], np.int32),
        ([-2147483649, 0, 2147483647], np.int64),
    ],
)
def test_numericaldowncaster_fit_selects_correct_integer_boundary_dtype(
    values, expected_dtype
):
    """Tests fit selects the smallest signed-integer dtype whose range covers the data.

    Args:
        values (list[int]): Boundary integer values to downcast.
        expected_dtype (type): Expected minimal-size integer dtype.

    Asserts:
        - The learned target dtype matches the expected minimal integer dtype
    """
    df = pd.DataFrame({"a": values})
    downcaster = NumericalDowncaster()

    downcaster.fit(df)

    if downcaster.dtype_map_["a"] != expected_dtype:
        raise AssertionError()


def test_numericaldowncaster_fit_respects_columns_subset():
    """Tests fit only learns dtypes for columns listed in the columns parameter.

    Asserts:
        - dtype_map_ contains an entry only for the subsetted column
        - Columns absent from the subset are excluded from dtype_map_
    """
    df = pd.DataFrame({"a": [1, 2, 3], "b": [100000, 200000, 300000]})
    downcaster = NumericalDowncaster(variables=["a"])

    downcaster.fit(df)

    if list(downcaster.dtype_map_.keys()) != ["a"]:
        raise AssertionError()


def test_numericaldowncaster_fit_downcasts_floats_only_when_lossless():
    """Tests the default lossless mode keeps float64 unless float32 represents every value.

    Asserts:
        - A column of exactly representable floats downcasts to float32
        - A column with 0.1-style values or values beyond float32 range stays float64
        - float_precision="float32" downcasts every float column
    """
    df = pd.DataFrame(
        {
            "exact": [0.5, 1.25, np.nan],
            "small": [0.1, 0.2, 0.3],
            "large": [1e300, -1e300, 2.5e250],
        }
    )
    lossless = NumericalDowncaster().fit(df)
    if lossless.dtype_map_["exact"] != np.float32:
        raise AssertionError()
    if lossless.dtype_map_["small"] != np.float64:
        raise AssertionError()
    if lossless.dtype_map_["large"] != np.float64:
        raise AssertionError()

    forced = NumericalDowncaster(float_precision="float32").fit(df)
    if not all(dtype == np.float32 for dtype in forced.dtype_map_.values()):
        raise AssertionError(forced.dtype_map_)


def test_numericaldowncaster_transform_applies_learned_dtypes():
    """Tests transform casts columns to the dtypes learned during fit.

    Asserts:
        - Integer column downcasts to int8 given its small value range
        - Float column downcasts to float32
    """
    df = pd.DataFrame({"small_int": [1, 2, 3], "any_float": [1.5, 2.5, 3.5]})
    downcaster = NumericalDowncaster()
    downcaster.fit(df)

    transformed = downcaster.transform(df)

    if transformed["small_int"].dtype != np.int8:
        raise AssertionError()
    if transformed["any_float"].dtype != np.float32:
        raise AssertionError()
    if not np.allclose(transformed["small_int"].to_numpy(), df["small_int"].to_numpy()):
        raise AssertionError()


def test_numericaldowncaster_transform_ignores_int_downcast_when_disabled():
    """Tests transform leaves integer columns unchanged when int_downcast is False.

    Asserts:
        - The learned dtype for the integer column equals its original dtype
    """
    df = pd.DataFrame({"a": pd.array([1, 2, 3], dtype="int64")})
    downcaster = NumericalDowncaster(int_downcast=False)

    downcaster.fit(df)

    if downcaster.dtype_map_["a"] != df["a"].dtype:
        raise AssertionError()


def test_numericaldowncaster_transform_ignores_float_downcast_when_disabled():
    """Tests transform leaves float columns unchanged when float_downcast is False.

    Asserts:
        - The learned dtype for the float column equals its original dtype
    """
    df = pd.DataFrame({"a": pd.array([1.5, 2.5, 3.5], dtype="float64")})
    downcaster = NumericalDowncaster(float_downcast=False)

    downcaster.fit(df)

    if downcaster.dtype_map_["a"] != df["a"].dtype:
        raise AssertionError()


def test_numericaldowncaster_keeps_large_integers_with_nan_exact():
    """Tests integer columns with NaN never go through float32 above 2**24.

    Asserts:
        - A nullable Int64 column with values > 2**24 and a missing value keeps exact values
        - An int column fitted clean and transformed with NaN keeps exact values
        - Small integers with NaN still become float32
    """
    big = [2**24 + 1, 2**31 + 7, None]
    df = pd.DataFrame({"id": pd.array(big, dtype="Int64")})
    out = NumericalDowncaster().fit_transform(df)
    if out["id"].iloc[:2].astype("int64").tolist() != big[:2] or not pd.isna(
        out["id"].iloc[2]
    ):
        raise AssertionError(out["id"].tolist())

    downcaster = NumericalDowncaster().fit(pd.DataFrame({"ts": [2**24 + 1, 2**30 + 3]}))
    out = downcaster.transform(pd.DataFrame({"ts": [2**24 + 1, np.nan]}))
    if out["ts"].iloc[0] != 2**24 + 1:
        raise AssertionError(out["ts"].tolist())

    small = NumericalDowncaster().fit(pd.DataFrame({"a": [1, 2, 3]}))
    if small.transform(pd.DataFrame({"a": [1.0, np.nan]}))["a"].dtype != np.float32:
        raise AssertionError()


def test_numericaldowncaster_transform_raises_not_fitted_before_fit():
    """Tests transform raises NotFittedError (not a NoneType error) before fit.

    Asserts:
        - NotFittedError is raised and dtype_map_ does not exist yet
    """
    df = pd.DataFrame({"a": [1, 2, 3]})
    downcaster = NumericalDowncaster()
    if hasattr(downcaster, "dtype_map_"):
        raise AssertionError()

    with pytest.raises(NotFittedError):
        downcaster.transform(df)


def test_numericaldowncaster_columns_alias_is_deprecated_but_works():
    """Tests the old columns= name still selects columns and emits a FutureWarning.

    Asserts:
        - Only the listed column is downcast
    """
    df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
    with pytest.warns(FutureWarning, match="variables"):
        downcaster = NumericalDowncaster(columns=["a"]).fit(df)
    if set(downcaster.dtype_map_) != {"a"}:
        raise AssertionError(downcaster.dtype_map_)


if __name__ == "__main__":
    print("Run this file with pytest to execute tests.")
