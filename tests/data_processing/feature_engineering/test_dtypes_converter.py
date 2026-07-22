"""Tests for kvbiii_ml.data_processing.feature_engineering.dtypes_converter module."""

import warnings

import numpy as np
import pandas as pd
import pytest

from kvbiii_ml.data_processing.feature_engineering.dtypes_converter import (
    DtypesConverter,
)


def test_dtypesconverter_init_defaults_dtype_map_to_none():
    """Tests DtypesConverter initialization with default parameters.

    Asserts:
        - dtype_map is None by default
        - dtype_map_ starts unset as None
    """
    converter = DtypesConverter()

    if converter.dtype_map is not None:
        raise AssertionError()
    if converter.dtype_map_ is not None:
        raise AssertionError()


def test_dtypesconverter_fit_learns_dtypes_when_map_is_none():
    """Tests fit learns per-column dtypes from the training DataFrame when dtype_map=None.

    Asserts:
        - dtype_map_ contains an entry for every column
        - Learned dtypes match the DataFrame's actual dtypes
        - fit returns self for chaining
    """
    df = pd.DataFrame({"a": [1, 2, 3], "b": [1.5, 2.5, 3.5], "c": ["x", "y", "z"]})
    converter = DtypesConverter()

    fitted = converter.fit(df)

    if fitted is not converter:
        raise AssertionError()
    if converter.dtype_map_ != {col: df[col].dtype for col in df.columns}:
        raise AssertionError()


def test_dtypesconverter_fit_keeps_explicit_map_when_provided():
    """Tests fit retains the user-supplied dtype_map unchanged in explicit-map mode.

    Asserts:
        - dtype_map_ equals the explicit dtype_map passed at construction
        - The learned per-column dtype inference is not used
    """
    df = pd.DataFrame({"a": [1, 2, 3], "b": [1.5, 2.5, 3.5]})
    explicit_map = {"a": "float32", "b": "float64"}
    converter = DtypesConverter(dtype_map=explicit_map)

    converter.fit(df)

    if converter.dtype_map_ != explicit_map:
        raise AssertionError()


def test_dtypesconverter_transform_converts_column_when_dtype_differs():
    """Tests transform converts a column and emits a UserWarning when dtypes differ.

    Asserts:
        - A UserWarning is raised mentioning the column name
        - The resulting column dtype matches the target dtype
        - Original DataFrame values are preserved
    """
    df = pd.DataFrame({"a": [1, 2, 3]})
    converter = DtypesConverter(dtype_map={"a": "float32"})
    converter.fit(df)

    with pytest.warns(UserWarning, match="Column 'a'"):
        transformed = converter.transform(df)

    if transformed["a"].dtype != np.float32:
        raise AssertionError()
    if not np.allclose(transformed["a"].to_numpy(), df["a"].to_numpy()):
        raise AssertionError()


def test_dtypesconverter_transform_skips_column_when_dtype_already_matches():
    """Tests transform leaves a column untouched when its dtype already matches the target.

    Asserts:
        - No UserWarning is emitted
        - The resulting DataFrame equals the input DataFrame
    """
    df = pd.DataFrame({"a": pd.array([1, 2, 3], dtype="int64")})
    converter = DtypesConverter(dtype_map={"a": "int64"})
    converter.fit(df)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        transformed = converter.transform(df)

    if caught:
        raise AssertionError(f"Unexpected warnings raised: {caught}")
    pd.testing.assert_frame_equal(transformed, df)


def test_dtypesconverter_transform_skips_column_not_in_dataframe():
    """Tests transform silently skips dtype_map entries absent from the input DataFrame.

    Asserts:
        - No exception is raised
        - The returned DataFrame keeps only the columns actually present
    """
    df = pd.DataFrame({"a": [1, 2, 3]})
    converter = DtypesConverter(dtype_map={"a": "int64", "missing_col": "float32"})
    converter.fit(df)

    transformed = converter.transform(df)

    if list(transformed.columns) != ["a"]:
        raise AssertionError()


def test_dtypesconverter_transform_warns_instead_of_raising_on_incompatible_cast():
    """Tests transform catches ValueError/TypeError from an incompatible astype call.

    Asserts:
        - Two UserWarnings are emitted: one for the conversion attempt, one for the failure
        - The column is left unconverted (original dtype and values preserved)
        - No exception propagates out of transform
    """
    df = pd.DataFrame({"a": ["not_a_number", "still_not", "nope"]})
    converter = DtypesConverter(dtype_map={"a": "int64"})
    converter.fit(df)

    with pytest.warns(UserWarning) as record:
        transformed = converter.transform(df)

    messages = [str(warning.message) for warning in record]
    if not any("Failed to convert column" in message for message in messages):
        raise AssertionError()
    if transformed["a"].dtype != df["a"].dtype:
        raise AssertionError()
    if not transformed["a"].equals(df["a"]):
        raise AssertionError()


def test_dtypesconverter_transform_raises_attributeerror_before_fit():
    """Tests transform raises AttributeError when called before fit.

    Asserts:
        - AttributeError is raised because dtype_map_ is still None
    """
    df = pd.DataFrame({"a": [1, 2, 3]})
    converter = DtypesConverter()

    with pytest.raises(AttributeError):
        converter.transform(df)


if __name__ == "__main__":
    print("Run this file with pytest to execute tests.")
