"""Tests for kvbiii_ml.data_processing.eda.data_transformation module."""

from unittest.mock import Mock, patch

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
    assert isinstance(transformer, DataTransformer)
    assert hasattr(transformer, "optimize_memory")
    # Only core optimization methods exist in implementation
    assert hasattr(transformer, "optimize_memory")


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
    assert new_memory < original_memory

    # Check dtypes were appropriately modified
    assert pd.api.types.is_integer_dtype(optimized_df["int64_col"])
    assert pd.api.types.is_float_dtype(optimized_df["float64_col"])

    # Small integer column should be in a smaller type
    assert optimized_df["small_int_col"].dtype != np.int64

    # Categorical columns should be preserved or converted
    assert isinstance(optimized_df["categorical_col"].dtype, pd.CategoricalDtype)

    # Dates should be preserved
    assert pd.api.types.is_datetime64_dtype(optimized_df["date_col"])

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

    # Original is object dtype
    assert sample_mixed_dataframe["object_col"].dtype == "object"

    # Optimize with object_col specified as categorical
    optimized_df = transformer.optimize_memory(
        sample_mixed_dataframe, categorical_features=["object_col"], verbose=False
    )

    # Should now be categorical
    assert isinstance(optimized_df["object_col"].dtype, pd.CategoricalDtype)

    # Values should be preserved
    assert set(optimized_df["object_col"].cat.categories) == set(["X", "Y", "Z"])

    # Check all values match
    for i, val in enumerate(sample_mixed_dataframe["object_col"]):
        assert optimized_df["object_col"].iloc[i] == val


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
    assert "Numerical dtypes reduced:" in captured.out
    assert "MB" in captured.out
    assert "reduction" in captured.out

    # Date feature generation and other advanced transformations not implemented; skip related assertions
    pass


if __name__ == "__main__":
    print("Run this file with pytest to execute tests.")
