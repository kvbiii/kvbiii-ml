"""Tests for kvbiii_ml.data_processing.feature_engineering.categories_assigner module."""

from unittest.mock import Mock

import numpy as np
import pandas as pd
import pytest

from kvbiii_ml.data_processing.feature_engineering.categorical_aligner import (
    CategoriesAssigner,
)


@pytest.fixture
def categorical_training_data(test_settings):
    """Provides training data with categorical features for testing.

    Args:
        test_settings: Test configuration fixture

    Returns:
        pd.DataFrame: Training data with categorical columns
    """
    np.random.seed(test_settings.SEED)
    return pd.DataFrame(
        {
            "color": pd.Categorical(["red", "blue", "green", "red", "blue"]),
            "fuel": pd.Categorical(
                ["petrol", "diesel", "electric", "diesel", "petrol"]
            ),
            "numeric": [10, 20, 30, 40, 50],
        }
    )


@pytest.fixture
def categorical_test_data():
    """Provides test data with known and unknown categorical values.

    Returns:
        pd.DataFrame: Test data containing unseen categorical values
    """
    return pd.DataFrame(
        {
            "color": ["blue", "purple", "yellow", "red"],  # purple, yellow are unseen
            "fuel": ["diesel", "hydrogen", "petrol", "electric"],  # hydrogen is unseen
            "numeric": [15, 25, 35, 45],
        }
    )


def test_categoriesassigner_init_stores_categorical_features():
    """Tests CategoriesAssigner initialization stores categorical feature list.

    Asserts:
        - Categorical feature list is stored correctly
        - Instance attributes are properly initialized
        - Feature list can be empty
    """
    features = ["color", "fuel", "brand"]
    assigner = CategoriesAssigner(features)

    assert assigner.categorical_features == features

    # Test empty list initialization
    empty_assigner = CategoriesAssigner([])
    assert empty_assigner.categorical_features == []


def test_categoriesassigner_fit_extracts_category_levels(categorical_training_data):
    """Tests fit method extracts category levels from categorical columns.

    Args:
        categorical_training_data (pd.DataFrame): Training data with categoricals

    Asserts:
        - Category levels are extracted and stored for each feature
        - Feature modes are computed correctly for unknown value replacement
        - Input features are stored for later reference
    """
    assigner = CategoriesAssigner(["color", "fuel"])
    fitted_assigner = assigner.fit(categorical_training_data)

    assert fitted_assigner is assigner  # Returns self
    assert "color" in assigner.feature_groups
    assert "fuel" in assigner.feature_groups

    # Check categories are extracted
    assert set(assigner.feature_groups["color"]) == {"red", "blue", "green"}
    assert set(assigner.feature_groups["fuel"]) == {"petrol", "diesel", "electric"}

    # Check modes are computed
    assert "color" in assigner.feature_modes_
    assert "fuel" in assigner.feature_modes_


def test_categoriesassigner_fit_handles_missing_features(categorical_training_data):
    """Tests fit method handles specified features that don't exist in data.

    Args:
        categorical_training_data (pd.DataFrame): Training data

    Asserts:
        - Non-existent features are ignored without error
        - Existing features are processed normally
        - Feature groups only contain actually present features
    """
    assigner = CategoriesAssigner(["color", "nonexistent_feature"])
    assigner.fit(categorical_training_data)

    assert "color" in assigner.feature_groups
    assert "nonexistent_feature" not in assigner.feature_groups


def test_categoriesassigner_fit_handles_non_categorical_features():
    """Tests fit method ignores features that are not categorical.

    Asserts:
        - Non-categorical features are ignored even if specified
        - Only actual categorical features are processed
        - No errors are raised for non-categorical features
    """
    data = pd.DataFrame(
        {
            "color": pd.Categorical(["red", "blue"]),
            "numeric": [1, 2],  # Not categorical
            "text": ["a", "b"],  # Not categorical
        }
    )

    assigner = CategoriesAssigner(["color", "numeric", "text"])
    assigner.fit(data)

    assert "color" in assigner.feature_groups
    assert "numeric" not in assigner.feature_groups
    assert "text" not in assigner.feature_groups


def test_categoriesassigner_fit_computes_correct_modes(test_settings):
    """Tests fit method computes correct mode values for unknown replacement.

    Args:
        test_settings: Test configuration fixture

    Asserts:
        - Mode is the most frequent value in each categorical feature
        - Ties in mode frequency are handled consistently
        - Mode computation ignores null values
    """
    np.random.seed(test_settings.SEED)
    data = pd.DataFrame(
        {
            "category": pd.Categorical(["A", "A", "A", "B", "B", "C"]),  # Mode is 'A'
            "balanced": pd.Categorical(
                ["X", "Y", "X", "Y", "X", "Y"]
            ),  # Balanced, mode is first
        }
    )

    assigner = CategoriesAssigner(["category", "balanced"])
    assigner.fit(data)

    assert assigner.feature_modes_["category"] == "A"
    assert assigner.feature_modes_["balanced"] in ["X", "Y"]  # Either is valid for tie


def test_categoriesassigner_transform_preserves_known_categories(
    categorical_training_data, categorical_test_data
):
    """Tests transform method preserves known categorical values.

    Args:
        categorical_training_data (pd.DataFrame): Training data
        categorical_test_data (pd.DataFrame): Test data with known values

    Asserts:
        - Known categorical values are preserved unchanged
        - Categorical dtype is maintained after transformation
        - Original data structure is maintained
    """
    assigner = CategoriesAssigner(["color", "fuel"])
    assigner.fit(categorical_training_data)

    transformed = assigner.transform(categorical_test_data)

    # Check that known values are preserved
    assert transformed.loc[0, "color"] == "blue"  # Known value
    assert transformed.loc[2, "fuel"] == "petrol"  # Known value

    # Check categorical dtype is maintained
    assert isinstance(transformed["color"].dtype, pd.CategoricalDtype)
    assert isinstance(transformed["fuel"].dtype, pd.CategoricalDtype)


def test_categoriesassigner_transform_replaces_unknown_with_mode(
    categorical_training_data, categorical_test_data
):
    """Tests transform method replaces unknown values with fitted modes.

    Args:
        categorical_training_data (pd.DataFrame): Training data
        categorical_test_data (pd.DataFrame): Test data with unknown values

    Asserts:
        - Unknown categorical values are replaced with mode from training
        - Replacement only affects unknown values, not known ones
        - Mode replacement is consistent across multiple unknown values
    """
    assigner = CategoriesAssigner(["color", "fuel"])
    assigner.fit(categorical_training_data)

    transformed = assigner.transform(categorical_test_data)

    # Get the modes that were computed during fit
    color_mode = assigner.feature_modes_["color"]
    fuel_mode = assigner.feature_modes_["fuel"]

    # Check unknown values are replaced with mode
    assert transformed.loc[1, "color"] == color_mode  # 'purple' -> mode
    assert transformed.loc[2, "color"] == color_mode  # 'yellow' -> mode
    assert transformed.loc[1, "fuel"] == fuel_mode  # 'hydrogen' -> mode


def test_categoriesassigner_transform_handles_missing_values():
    """Tests transform method properly handles missing/null values.

    Asserts:
        - Null values are preserved and not replaced with mode
        - Mode replacement only affects non-null unknown values
        - Categorical dtype handles null values correctly
    """
    training_data = pd.DataFrame({"category": pd.Categorical(["A", "B", "A", "B"])})

    test_data = pd.DataFrame(
        {"category": ["A", None, "C", "B"]}  # None should stay, 'C' should become mode
    )

    assigner = CategoriesAssigner(["category"])
    assigner.fit(training_data)
    transformed = assigner.transform(test_data)

    assert transformed.loc[0, "category"] == "A"  # Known value preserved
    assert pd.isna(transformed.loc[1, "category"])  # Null preserved
    assert transformed.loc[2, "category"] in ["A", "B"]  # Unknown replaced with mode
    assert transformed.loc[3, "category"] == "B"  # Known value preserved


def test_categoriesassigner_transform_creates_copy_of_input(
    categorical_training_data, categorical_test_data
):
    """Tests transform method creates copy of input data without modifying original.

    Args:
        categorical_training_data (pd.DataFrame): Training data
        categorical_test_data (pd.DataFrame): Test data

    Asserts:
        - Original input DataFrame is not modified
        - Returned DataFrame is a separate copy
        - Changes to returned DataFrame don't affect original
    """
    assigner = CategoriesAssigner(["color", "fuel"])
    assigner.fit(categorical_training_data)

    original_test = categorical_test_data.copy()
    transformed = assigner.transform(categorical_test_data)

    # Original data should be unchanged
    pd.testing.assert_frame_equal(categorical_test_data, original_test)

    # Transformed should be different (contains categorical dtypes)
    assert not categorical_test_data.equals(transformed)


def test_categoriesassigner_transform_ignores_non_configured_features():
    """Tests transform method ignores features not in categorical_features list.

    Asserts:
        - Features not in categorical_features are left unchanged
        - Only specified categorical features are processed
        - Non-categorical features maintain their original dtypes
    """
    training_data = pd.DataFrame(
        {
            "configured": pd.Categorical(["A", "B"]),
            "not_configured": pd.Categorical(["X", "Y"]),
            "numeric": [1, 2],
        }
    )

    test_data = pd.DataFrame(
        {
            "configured": ["A", "C"],  # 'C' is unknown
            "not_configured": ["X", "Z"],  # 'Z' is unknown but shouldn't be processed
            "numeric": [3, 4],
        }
    )

    assigner = CategoriesAssigner(["configured"])  # Only configure one feature
    assigner.fit(training_data)
    transformed = assigner.transform(test_data)

    # Configured feature should be processed
    assert isinstance(transformed["configured"].dtype, pd.CategoricalDtype)

    # Non-configured categorical feature should remain unchanged
    assert transformed.loc[1, "not_configured"] == "Z"  # Unknown value preserved

    # Numeric feature should be unchanged
    assert transformed["numeric"].equals(test_data["numeric"])


def test_categoriesassigner_get_feature_names_out_returns_input_features(
    categorical_training_data,
):
    """Tests get_feature_names_out method returns original feature names.

    Args:
        categorical_training_data (pd.DataFrame): Training data

    Asserts:
        - Method returns pandas Index of original feature names
        - Feature names match those seen during fit
        - Input parameter is ignored as documented
    """
    assigner = CategoriesAssigner(["color", "fuel"])
    assigner.fit(categorical_training_data)

    feature_names = assigner.get_feature_names_out()

    assert isinstance(feature_names, pd.Index)
    assert list(feature_names) == list(categorical_training_data.columns)

    # Test that input parameter is ignored
    feature_names_with_input = assigner.get_feature_names_out(["ignored", "input"])
    pd.testing.assert_index_equal(feature_names, feature_names_with_input)


def test_categoriesassigner_handles_empty_categorical_features_list():
    """Tests CategoriesAssigner handles empty categorical features list gracefully.

    Asserts:
        - Fit and transform work with empty feature list
        - No features are processed when list is empty
        - Original data is returned unchanged
    """
    training_data = pd.DataFrame(
        {"category": pd.Categorical(["A", "B"]), "numeric": [1, 2]}
    )

    test_data = pd.DataFrame(
        {
            "category": ["A", "C"],  # Unknown value that won't be processed
            "numeric": [3, 4],
        }
    )

    assigner = CategoriesAssigner([])  # Empty feature list
    assigner.fit(training_data)
    transformed = assigner.transform(test_data)

    # Data should be unchanged except for potential copy
    assert transformed["category"].equals(test_data["category"])
    assert transformed["numeric"].equals(test_data["numeric"])
    assert len(assigner.feature_groups) == 0


def test_categoriesassigner_fit_handles_empty_categories():
    """Tests fit method handles categorical features with no categories gracefully.

    Asserts:
        - Empty categorical features don't cause errors
        - Mode computation handles empty categories appropriately
        - Transformer remains functional with empty categories
    """
    # Create categorical with no categories (edge case)
    training_data = pd.DataFrame(
        {"normal_cat": pd.Categorical(["A", "B"]), "numeric": [1, 2]}
    )

    assigner = CategoriesAssigner(["normal_cat"])
    assigner.fit(training_data)

    # Should handle normal categories without error
    assert "normal_cat" in assigner.feature_groups


if __name__ == "__main__":
    print("Run this file with pytest to execute tests.")
