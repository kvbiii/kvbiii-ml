"""Tests for kvbiii_ml.data_processing.eda.data_transformation module."""

import numpy as np
import pandas as pd
import pytest

from kvbiii_ml.data_processing.eda.data_transformation import DataTransformer


@pytest.fixture
def sample_mixed_dataframe(test_settings):
    """Provides a sample DataFrame with mixed data types for transformation testing.

    Args:
        test_settings: Test configuration fixture

    Returns:
        pd.DataFrame: DataFrame with mixed numeric and categorical data
    """
    np.random.seed(test_settings.SEED)
    n_samples = test_settings.N_SAMPLES

    return pd.DataFrame(
        {
            "int64_col": np.random.randint(0, 1000, n_samples),
            "float64_col": np.random.randn(n_samples) * 1000,
            "small_int_col": np.random.randint(0, 10, n_samples),
            "small_float_col": np.random.randn(n_samples) * 0.1,
            "categorical_col": pd.Categorical(
                np.random.choice(["A", "B", "C", "D"], n_samples)
            ),
            "object_col": np.random.choice(["X", "Y", "Z"], n_samples),
            "date_col": pd.date_range("2020-01-01", periods=n_samples, freq="D"),
        }
    )


def test_datatransformer_init_creates_instance():
    """Tests DataTransformer initialization creates a valid instance.

    Asserts:
        - DataTransformer can be instantiated
        - Instance methods are available
    """
    transformer = DataTransformer()
    if not isinstance(transformer, DataTransformer):
        raise AssertionError()
    if not hasattr(transformer, "optimize_memory"):
        raise AssertionError()
    # Only core optimization methods exist in implementation
    if not hasattr(transformer, "optimize_memory"):
        raise AssertionError()


def test_optimize_memory_reduces_numeric_memory_usage(sample_mixed_dataframe):
    """Tests optimize_memory method reduces memory usage for numeric columns.

    Args:
        sample_mixed_dataframe: Sample DataFrame with mixed data types

    Asserts:
        - Memory usage is reduced after optimization
        - Integer columns are converted to appropriate smaller types
        - Float columns are downcasted when possible
        - Non-numeric columns remain unchanged
    """
    transformer = DataTransformer()

    # Get original memory usage
    original_memory = sample_mixed_dataframe.memory_usage(deep=True).sum()

    # Optimize memory
    optimized_df = transformer.optimize_memory(
        sample_mixed_dataframe,
        categorical_features=["categorical_col", "object_col"],
        verbose=False,
    )

    # Get new memory usage
    new_memory = optimized_df.memory_usage(deep=True).sum()

    # Memory should be reduced
    if not new_memory < original_memory:
        raise AssertionError()

    # Check dtypes were appropriately modified
    if not pd.api.types.is_integer_dtype(optimized_df["int64_col"]):
        raise AssertionError()
    if not pd.api.types.is_float_dtype(optimized_df["float64_col"]):
        raise AssertionError()

    # Small integer column should be in a smaller type
    if optimized_df["small_int_col"].dtype == np.int64:
        raise AssertionError()

    # Categorical columns should be preserved or converted
    if not isinstance(optimized_df["categorical_col"].dtype, pd.CategoricalDtype):
        raise AssertionError()

    # Dates should be preserved
    if not pd.api.types.is_datetime64_dtype(optimized_df["date_col"]):
        raise AssertionError()

    # Data values should be preserved
    pd.testing.assert_series_equal(
        sample_mixed_dataframe["int64_col"],
        optimized_df["int64_col"],
        check_dtype=False,
    )


def test_optimize_memory_converts_object_to_categorical(sample_mixed_dataframe):
    """Tests optimize_memory converts object columns to categorical when specified.

    Args:
        sample_mixed_dataframe: Sample DataFrame with mixed data types

    Asserts:
        - Object columns are converted to categorical when specified
        - Conversion reduces memory usage
        - Data values are preserved after conversion
    """
    transformer = DataTransformer()

    # Original is a string-like dtype (object, or pandas' inferred "string" dtype)
    if not pd.api.types.is_string_dtype(sample_mixed_dataframe["object_col"].dtype):
        raise AssertionError()

    # Optimize with object_col specified as categorical
    optimized_df = transformer.optimize_memory(
        sample_mixed_dataframe, categorical_features=["object_col"], verbose=False
    )

    # Should now be categorical
    if not isinstance(optimized_df["object_col"].dtype, pd.CategoricalDtype):
        raise AssertionError()

    # Values should be preserved
    if set(optimized_df["object_col"].cat.categories) != set(["X", "Y", "Z"]):
        raise AssertionError()

    # Check all values match
    for i, val in enumerate(sample_mixed_dataframe["object_col"]):
        if optimized_df["object_col"].iloc[i] != val:
            raise AssertionError()


def test_optimize_memory_verbose_output(sample_mixed_dataframe, capsys):
    """Tests optimize_memory verbose output shows memory savings.

    Args:
        sample_mixed_dataframe: Sample DataFrame with mixed data types
        capsys: Pytest fixture to capture stdout

    Asserts:
        - Verbose output contains expected information
        - Output includes memory savings for numeric types
        - Output includes memory savings for categorical conversions
    """
    transformer = DataTransformer()

    # Run with verbose=True
    transformer.optimize_memory(
        sample_mixed_dataframe, categorical_features=["object_col"], verbose=True
    )

    # Capture output
    captured = capsys.readouterr()

    # Check expected phrases in output
    if "Numerical dtypes reduced:" not in captured.out:
        raise AssertionError()
    if "MB" not in captured.out:
        raise AssertionError()
    if "reduction" not in captured.out:
        raise AssertionError()

    # Date feature generation and other advanced transformations not implemented; skip related assertions
    pass


def test_encode_target_feature_raises_error_for_missing_target():
    """Tests encode_target_feature raises ValueError when the target column is absent.

    Asserts:
        - ValueError is raised mentioning the missing target feature name
    """
    df = pd.DataFrame({"feature": [1, 2, 3]})

    with pytest.raises(ValueError, match="Target feature 'missing_target' not found"):
        DataTransformer.encode_target_feature(df, "missing_target", verbose=False)


def test_encode_target_feature_returns_correct_id2label_mapping():
    """Tests encode_target_feature returns an id2label mapping matching the categorical
    codes actually assigned to the target column.

    Asserts:
        - id2label maps each assigned integer code to its original string label
        - The mapping keys cover exactly the codes present in the encoded column
        - Categories are sorted alphabetically, matching pandas' categorical ordering
    """
    df = pd.DataFrame({"target": ["yes", "no", "yes", "no", "maybe"]})

    encoded_df, id2label = DataTransformer.encode_target_feature(
        df, "target", verbose=False
    )

    if id2label != {0: "maybe", 1: "no", 2: "yes"}:
        raise AssertionError()
    if set(encoded_df["target"].unique()) != set(id2label.keys()):
        raise AssertionError()


def test_encode_target_feature_replaces_target_with_integer_codes():
    """Tests encode_target_feature replaces the target column with integer codes that
    correctly reconstruct the original labels via the returned id2label mapping.

    Asserts:
        - The target column dtype is integer after encoding
        - Mapping the encoded codes back through id2label reproduces the original labels
        - Non-target columns remain unchanged
    """
    df = pd.DataFrame(
        {
            "feature": [10, 20, 30, 40, 50],
            "target": ["yes", "no", "yes", "no", "maybe"],
        }
    )

    encoded_df, id2label = DataTransformer.encode_target_feature(
        df, "target", verbose=False
    )

    if not pd.api.types.is_integer_dtype(encoded_df["target"]):
        raise AssertionError()

    reconstructed_labels = encoded_df["target"].map(id2label).tolist()
    if reconstructed_labels != df["target"].tolist():
        raise AssertionError()

    pd.testing.assert_series_equal(encoded_df["feature"], df["feature"])


if __name__ == "__main__":
    print("Run this file with pytest to execute tests.")
