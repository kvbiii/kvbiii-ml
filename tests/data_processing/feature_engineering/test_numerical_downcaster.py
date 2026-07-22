"""Tests for kvbiii_ml.data_processing.feature_engineering.numerical_downcaster module."""

import numpy as np
import pandas as pd
import pytest

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
    downcaster = NumericalDowncaster(columns=["a"])

    downcaster.fit(df)

    if list(downcaster.dtype_map_.keys()) != ["a"]:
        raise AssertionError()


def test_numericaldowncaster_fit_downcasts_float_columns_to_float32_regardless_of_range():
    """Tests fit always assigns float32 to float columns, independent of value magnitude.

    Asserts:
        - A small-range float column downcasts to float32
        - A large-range float column also downcasts to float32
    """
    df = pd.DataFrame({"small": [0.1, 0.2, 0.3], "large": [1e300, -1e300, 2.5e250]})
    downcaster = NumericalDowncaster()

    downcaster.fit(df)

    if downcaster.dtype_map_["small"] != np.float32:
        raise AssertionError()
    if downcaster.dtype_map_["large"] != np.float32:
        raise AssertionError()


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


def test_numericaldowncaster_transform_warns_instead_of_raising_on_incompatible_cast():
    """Tests transform catches ValueError/TypeError from an incompatible astype call.

    Asserts:
        - A UserWarning is emitted describing the failed downcast
        - The column is left unconverted on the incompatible DataFrame
        - No exception propagates out of transform
    """
    train_df = pd.DataFrame({"a": [1, 2, 3]})
    downcaster = NumericalDowncaster()
    downcaster.fit(train_df)

    incompatible_df = pd.DataFrame({"a": [1.0, np.nan, 3.0]})

    with pytest.warns(UserWarning, match="Failed to downcast column"):
        transformed = downcaster.transform(incompatible_df)

    if transformed["a"].dtype != incompatible_df["a"].dtype:
        raise AssertionError()
    if not transformed["a"].equals(incompatible_df["a"]):
        raise AssertionError()


def test_numericaldowncaster_transform_raises_attributeerror_before_fit():
    """Tests transform raises AttributeError when called before fit.

    Asserts:
        - AttributeError is raised because dtype_map_ is still None
    """
    df = pd.DataFrame({"a": [1, 2, 3]})
    downcaster = NumericalDowncaster()

    with pytest.raises(AttributeError):
        downcaster.transform(df)


if __name__ == "__main__":
    print("Run this file with pytest to execute tests.")
