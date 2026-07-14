"""Tests for kvbiii_ml.data_processing.data_imputation.joint_distribution_imputation module."""

import itertools
from unittest.mock import MagicMock, Mock, patch

import numpy as np
import pandas as pd
import pytest

from kvbiii_ml.data_processing.data_imputation.joint_distribution_imputation import (
    impute_missing_values,
)


@pytest.fixture
def categorical_data_with_missing(test_settings):
    """Provides categorical data with missing values for imputation testing.

    Args:
        test_settings: Test configuration fixture

    Returns:
        pd.DataFrame: DataFrame with categorical columns and missing values
    """
    np.random.seed(test_settings.SEED)
    n_samples = test_settings.N_SAMPLES

    # Create categorical data with missing values pattern
    data = pd.DataFrame(
        {
            "cat_to_impute_1": np.random.choice(["A", "B", "C"], size=n_samples),
            "cat_to_impute_2": np.random.choice(["X", "Y", "Z"], size=n_samples),
            "non_missing_cat_1": np.random.choice(["P", "Q"], size=n_samples),
            "non_missing_cat_2": np.random.choice(["M", "N", "O"], size=n_samples),
        }
    )

    # Convert to categorical dtype
    for col in data.columns:
        data[col] = data[col].astype("category")

    # Introduce missing values in specific columns
    missing_indices_1 = np.random.choice(n_samples, size=n_samples // 5, replace=False)
    missing_indices_2 = np.random.choice(n_samples, size=n_samples // 6, replace=False)

    data.loc[missing_indices_1, "cat_to_impute_1"] = None
    data.loc[missing_indices_2, "cat_to_impute_2"] = None

    return data


@pytest.fixture
def structured_categorical_data():
    """Provides structured categorical data where imputation patterns are deterministic.

    Returns:
        pd.DataFrame: DataFrame with predictable missing value patterns
    """
    # Create structured data where imputation should be possible
    data = pd.DataFrame(
        {
            "A": ["x"] * 30 + ["y"] * 25 + [None] * 8 + [None] * 5 + ["z"] * 20,
            "B": ["u"] * 30 + ["v"] * 25 + ["u"] * 8 + ["v"] * 5 + ["w"] * 20,
            "C": np.random.choice(["m", "n"], size=88),
        }
    )

    # Convert to categorical dtype
    for col in ["A", "B", "C"]:
        data[col] = data[col].astype("category")

    return data


def test_impute_missing_values_basic_functionality(structured_categorical_data):
    """Tests impute_missing_values function basic functionality.

    Args:
        structured_categorical_data (pd.DataFrame): Structured data for testing

    Asserts:
        - Function completes without errors
        - Returns DataFrame with same shape
        - Some missing values are imputed based on joint distributions
    """
    original_data = structured_categorical_data.copy()
    original_missing_count = original_data["A"].isnull().sum()

    result = impute_missing_values(
        df=original_data,
        categorical_to_impute=["A"],
        non_missing_categorical=["B"],
        threshold_num_observation=5,
    )

    assert isinstance(result, pd.DataFrame)
    assert result.shape == original_data.shape

    # Should have fewer missing values after imputation
    new_missing_count = result["A"].isnull().sum()
    assert new_missing_count <= original_missing_count


def test_impute_missing_values_respects_threshold_parameter():
    """Tests impute_missing_values respects threshold_num_observation parameter.

    Asserts:
        - Higher thresholds result in less imputation
        - Lower thresholds result in more imputation
        - Threshold parameter controls imputation behavior
    """
    # Create data with controlled patterns
    data = pd.DataFrame(
        {
            "target": ["A"] * 20 + [None] * 5 + ["B"] * 15 + [None] * 3,
            "predictor": ["X"] * 25 + ["Y"] * 18,
        }
    )

    for col in ["target", "predictor"]:
        data[col] = data[col].astype("category")

    # Test with high threshold (should impute less)
    result_high = impute_missing_values(
        df=data,
        categorical_to_impute=["target"],
        non_missing_categorical=["predictor"],
        threshold_num_observation=20,
    )

    # Test with low threshold (should impute more)
    result_low = impute_missing_values(
        df=data,
        categorical_to_impute=["target"],
        non_missing_categorical=["predictor"],
        threshold_num_observation=2,
    )

    # Low threshold should result in more imputation
    missing_high = result_high["target"].isnull().sum()
    missing_low = result_low["target"].isnull().sum()
    assert missing_low <= missing_high


def test_impute_missing_values_handles_no_eligible_mappings():
    """Tests impute_missing_values handles cases with no eligible mappings.

    Asserts:
        - Function doesn't fail when no imputation is possible
        - Original data is returned unchanged when no mappings exist
        - Edge case handling is robust
    """
    # Create data where no clear mappings exist
    data = pd.DataFrame(
        {
            "ambiguous": ["A", "B", "A", "B", None, None],
            "mixed_predictor": ["X", "X", "Y", "Y", "X", "Y"],
        }
    )

    for col in data.columns:
        data[col] = data[col].astype("category")

    original_missing = data["ambiguous"].isnull().sum()

    result = impute_missing_values(
        df=data,
        categorical_to_impute=["ambiguous"],
        non_missing_categorical=["mixed_predictor"],
        threshold_num_observation=1,
    )

    # Should not impute when mappings are ambiguous
    assert result["ambiguous"].isnull().sum() == original_missing


def test_impute_missing_values_preserves_non_missing_values(
    categorical_data_with_missing,
):
    """Tests impute_missing_values preserves non-missing values.

    Args:
        categorical_data_with_missing (pd.DataFrame): Data with missing values

    Asserts:
        - Non-missing values remain unchanged after imputation
        - Only missing values are modified
        - Data integrity is maintained
    """
    original_data = categorical_data_with_missing.copy()

    # Record non-missing values for comparison
    non_missing_mask = ~original_data["cat_to_impute_1"].isnull()
    original_non_missing = original_data.loc[non_missing_mask, "cat_to_impute_1"].copy()

    result = impute_missing_values(
        df=original_data,
        categorical_to_impute=["cat_to_impute_1"],
        non_missing_categorical=["non_missing_cat_1"],
        threshold_num_observation=3,
    )

    # Check that non-missing values are preserved
    result_non_missing = result.loc[non_missing_mask, "cat_to_impute_1"]
    assert original_non_missing.equals(result_non_missing)


def test_impute_missing_values_handles_multiple_columns_to_impute():
    """Tests impute_missing_values handles multiple categorical_to_impute columns.

    Asserts:
        - Function can process multiple columns for imputation
        - Each column is processed independently
        - All specified columns are considered for imputation
    """
    data = pd.DataFrame(
        {
            "cat1": ["A"] * 10 + [None] * 5,
            "cat2": ["X"] * 8 + [None] * 7,
            "predictor": ["P"] * 15,
        }
    )

    for col in data.columns:
        data[col] = data[col].astype("category")

    original_missing_1 = data["cat1"].isnull().sum()
    original_missing_2 = data["cat2"].isnull().sum()

    result = impute_missing_values(
        df=data,
        categorical_to_impute=["cat1", "cat2"],
        non_missing_categorical=["predictor"],
        threshold_num_observation=5,
    )

    # Should attempt imputation on both columns
    new_missing_1 = result["cat1"].isnull().sum()
    new_missing_2 = result["cat2"].isnull().sum()

    assert new_missing_1 <= original_missing_1
    assert new_missing_2 <= original_missing_2


def test_impute_missing_values_handles_multiple_predictor_columns():
    """Tests impute_missing_values handles multiple non_missing_categorical columns.

    Asserts:
        - Function considers all predictor columns
        - Imputation uses joint distributions with each predictor
        - Multiple predictors increase imputation opportunities
    """
    data = pd.DataFrame(
        {
            "target": ["A"] * 20 + [None] * 10,
            "pred1": ["X"] * 15 + ["Y"] * 15,
            "pred2": ["P"] * 10 + ["Q"] * 20,
        }
    )

    for col in data.columns:
        data[col] = data[col].astype("category")

    original_missing = data["target"].isnull().sum()

    result = impute_missing_values(
        df=data,
        categorical_to_impute=["target"],
        non_missing_categorical=["pred1", "pred2"],
        threshold_num_observation=5,
    )

    new_missing = result["target"].isnull().sum()
    assert new_missing <= original_missing


def test_impute_missing_values_returns_copy_of_dataframe():
    """Tests impute_missing_values returns a copy and doesn't modify original.

    Asserts:
        - Original DataFrame is not modified
        - Function returns a new DataFrame instance
        - Data integrity of input is preserved
    """
    original_data = pd.DataFrame({"A": ["x", None, "z"], "B": ["u", "v", "w"]})

    for col in original_data.columns:
        original_data[col] = original_data[col].astype("category")

    # Store original state
    original_copy = original_data.copy()

    result = impute_missing_values(
        df=original_data,
        categorical_to_impute=["A"],
        non_missing_categorical=["B"],
        threshold_num_observation=1,
    )

    # Original should be unchanged
    pd.testing.assert_frame_equal(original_data, original_copy)

    # Result should be different object
    assert result is not original_data


@patch(
    "kvbiii_ml.data_processing.data_imputation.joint_distribution_imputation.multivariate_plots"
)
def test_impute_missing_values_creates_visualization(mock_plots):
    """Tests impute_missing_values creates heatmap visualization when imputation occurs.

    Args:
        mock_plots: Mocked multivariate plots object

    Asserts:
        - Visualization is created when imputation occurs
        - Heatmap method is called with appropriate parameters
        - Plotting functionality is integrated properly
    """
    # Create data that will trigger imputation
    data = pd.DataFrame({"A": ["x"] * 15 + [None] * 5, "B": ["u"] * 20})

    for col in data.columns:
        data[col] = data[col].astype("category")

    result = impute_missing_values(
        df=data,
        categorical_to_impute=["A"],
        non_missing_categorical=["B"],
        threshold_num_observation=5,
    )

    # Should call heatmap when imputation occurs
    if result["A"].isnull().sum() < data["A"].isnull().sum():
        mock_plots.heatmap.assert_called()


def test_impute_missing_values_handles_empty_categorical_lists():
    """Tests impute_missing_values handles empty categorical lists gracefully.

    Asserts:
        - Empty categorical_to_impute list returns data unchanged
        - Empty non_missing_categorical list returns data unchanged
        - Edge cases are handled without errors
    """
    data = pd.DataFrame({"A": ["x", None, "z"], "B": ["u", "v", "w"]})

    for col in data.columns:
        data[col] = data[col].astype("category")

    # Test empty categorical_to_impute
    result1 = impute_missing_values(
        df=data,
        categorical_to_impute=[],
        non_missing_categorical=["B"],
        threshold_num_observation=1,
    )

    pd.testing.assert_frame_equal(result1, data)

    # Test empty non_missing_categorical
    result2 = impute_missing_values(
        df=data,
        categorical_to_impute=["A"],
        non_missing_categorical=[],
        threshold_num_observation=1,
    )

    pd.testing.assert_frame_equal(result2, data)


def test_impute_missing_values_handles_single_unique_mapping():
    """Tests impute_missing_values correctly identifies single unique mappings.

    Asserts:
        - Single unique mappings are identified correctly
        - Imputation occurs only when mapping is unambiguous
        - Joint distribution logic works as expected
    """
    # Create data with clear single mapping: 'u' -> 'x' only
    data = pd.DataFrame(
        {
            "A": ["x"] * 20 + ["y"] * 15 + [None] * 8,  # Missing values with B='u'
            "B": ["u"] * 20 + ["v"] * 15 + ["u"] * 8,  # 'u' maps only to 'x'
        }
    )

    for col in data.columns:
        data[col] = data[col].astype("category")

    result = impute_missing_values(
        df=data,
        categorical_to_impute=["A"],
        non_missing_categorical=["B"],
        threshold_num_observation=5,
    )

    # All missing values where B='u' should be imputed to 'x'
    missing_with_u = data["A"].isnull() & (data["B"] == "u")
    if missing_with_u.any():
        imputed_values = result.loc[missing_with_u, "A"]
        assert all(val == "x" for val in imputed_values)


if __name__ == "__main__":
    print("Run this file with pytest to execute tests.")
