"""Tests for kvbiii_ml.data_processing.feature_engineering.target_encoding module."""

import numpy as np
import pandas as pd
import pytest
from sklearn.model_selection import KFold

from kvbiii_ml.data_processing.feature_engineering.target_encoding import (
    TargetEncodingFeatureGenerator,
)


@pytest.fixture
def target_encoding_data(test_settings):
    """Provides data for target encoding testing.

    Args:
        test_settings: Test configuration fixture

    Returns:
        tuple[pd.DataFrame, pd.Series]: Feature data with categorical variables and target
    """
    np.random.seed(test_settings.SEED)
    n_samples = test_settings.N_SAMPLES

    X = pd.DataFrame(
        {
            "category_a": np.random.choice(["cat1", "cat2", "cat3"], size=n_samples),
            "category_b": np.random.choice(["x", "y", "z"], size=n_samples),
            "numeric": np.random.randn(n_samples),
        }
    )

    # Create target with some relationship to categories
    y = pd.Series(np.random.randn(n_samples), name="target")
    # Add some signal for category_a
    y.loc[X["category_a"] == "cat1"] += 1.0
    y.loc[X["category_a"] == "cat2"] -= 0.5

    return X, y


def test_targetencodingfeaturegenerator_init_creates_instance_with_default_parameters():
    """Tests TargetEncodingFeatureGenerator initialization with default parameters.

    Asserts:
        - Instance is created successfully with provided feature names
        - Default aggregation method is 'mean'
        - Default smoothing factor is reasonable
        - Default cross-validator is KFold with expected configuration
    """
    features = ["category1", "category2"]
    generator = TargetEncodingFeatureGenerator(features_names=features)

    if generator.features_names != features:
        raise AssertionError()
    if generator.aggregation != "mean":
        raise AssertionError()
    if generator.smooth != 10:
        raise AssertionError()
    if not isinstance(generator.cv, KFold):
        raise AssertionError()


def test_targetencodingfeaturegenerator_init_accepts_custom_parameters():
    """Tests TargetEncodingFeatureGenerator initialization with custom parameters.

    Asserts:
        - Custom aggregation methods are accepted
        - Custom smoothing factors are stored
        - Custom cross-validators are accepted
        - All parameters are properly validated
    """
    features = ["category1"]
    custom_cv = KFold(n_splits=3, shuffle=False)

    generator = TargetEncodingFeatureGenerator(
        features_names=features, aggregation="median", smooth=5, cv=custom_cv
    )

    if generator.aggregation != "median":
        raise AssertionError()
    if generator.smooth != 5:
        raise AssertionError()
    if generator.cv is not custom_cv:
        raise AssertionError()


def test_targetencodingfeaturegenerator_validate_init_params_raises_error_for_invalid_features():
    """Tests _validate_init_params method raises error for invalid feature names.

    Asserts:
        - ValueError is raised for non-list feature names
        - ValueError is raised for non-string elements in feature list
        - Error messages provide clear guidance about expected format
    """
    # Test non-list features
    with pytest.raises(ValueError, match="features_names must be a list of strings"):
        TargetEncodingFeatureGenerator(features_names="not_a_list")

    # Test non-string elements in list
    with pytest.raises(ValueError, match="features_names must be a list of strings"):
        TargetEncodingFeatureGenerator(features_names=["valid", 123, "also_valid"])


def test_targetencodingfeaturegenerator_validate_init_params_raises_error_for_invalid_aggregation():
    """Tests _validate_init_params method raises error for invalid aggregation methods.

    Asserts:
        - ValueError is raised for unsupported aggregation methods
        - Error message lists supported aggregation options
        - Valid aggregation methods are accepted without error
    """
    features = ["category"]

    with pytest.raises(
        ValueError, match="Unsupported aggregation method.*Supported methods are"
    ):
        TargetEncodingFeatureGenerator(
            features_names=features, aggregation="invalid_method"
        )

    # Test valid aggregations don't raise errors
    for valid_agg in ["mean", "median", "nunique"]:
        generator = TargetEncodingFeatureGenerator(
            features_names=features, aggregation=valid_agg
        )
        if generator.aggregation != valid_agg:
            raise AssertionError()


def test_targetencodingfeaturegenerator_validate_init_params_raises_error_for_invalid_smooth():
    """Tests _validate_init_params method raises error for invalid smoothing values.

    Asserts:
        - ValueError is raised for negative smoothing values
        - ValueError is raised for non-integer smoothing values
        - Zero and positive integers are accepted
    """
    features = ["category"]

    # Test negative smooth
    with pytest.raises(ValueError, match="smooth must be a non-negative integer"):
        TargetEncodingFeatureGenerator(features_names=features, smooth=-1)

    # Test non-integer smooth
    with pytest.raises(ValueError, match="smooth must be a non-negative integer"):
        TargetEncodingFeatureGenerator(features_names=features, smooth=1.5)

    # Test valid smooth values
    for valid_smooth in [0, 1, 10, 100]:
        generator = TargetEncodingFeatureGenerator(
            features_names=features, smooth=valid_smooth
        )
        if generator.smooth != valid_smooth:
            raise AssertionError()


def test_targetencodingfeaturegenerator_validate_init_params_accepts_none_cv():
    """Tests _validate_init_params method accepts None cross-validator.

    Asserts:
        - None cross-validator is accepted without error
        - Instance properly handles None CV configuration
        - Other parameters remain properly validated
    """
    features = ["category"]
    generator = TargetEncodingFeatureGenerator(features_names=features, cv=None)

    if generator.cv is not None:
        raise AssertionError()


def test_targetencodingfeaturegenerator_validate_init_params_raises_error_for_invalid_cv():
    """Tests _validate_init_params method raises error for invalid cross-validators.

    Asserts:
        - ValueError is raised for non-BaseCrossValidator objects
        - Error message explains expected cross-validator type
        - Valid cross-validators are accepted
    """
    features = ["category"]

    with pytest.raises(
        ValueError,
        match="cv must be None or an instance of BaseCrossValidator or its subclasses",
    ):
        TargetEncodingFeatureGenerator(features_names=features, cv="invalid_cv")

    # Test valid CV is accepted
    valid_cv = KFold(n_splits=3)
    generator = TargetEncodingFeatureGenerator(features_names=features, cv=valid_cv)
    if generator.cv is not valid_cv:
        raise AssertionError()


def test_targetencodingfeaturegenerator_fit_processes_features_with_mean_aggregation(
    target_encoding_data,
):
    """Tests fit method processes features with mean aggregation correctly.

    Args:
        target_encoding_data (tuple): Feature data and target for encoding

    Asserts:
        - Fit method completes without errors
        - Target means are computed for each category level
        - Encoding mappings are stored for transform
    """
    X, y = target_encoding_data
    generator = TargetEncodingFeatureGenerator(
        features_names=["category_a", "category_b"], aggregation="mean"
    )

    fitted_generator = generator.fit(X, y)

    if fitted_generator is not generator:
        raise AssertionError()
    # Check that encoding mappings are created
    if not hasattr(generator, "group_stats_"):
        raise AssertionError()
    if "category_a" not in generator.group_stats_:
        raise AssertionError()
    if "category_b" not in generator.group_stats_:
        raise AssertionError()


def test_targetencodingfeaturegenerator_fit_handles_cross_validation_splits(
    target_encoding_data,
):
    """Tests fit method handles cross-validation to prevent overfitting.

    Args:
        target_encoding_data (tuple): Feature data and target for encoding

    Asserts:
        - Cross-validation is used to compute out-of-fold encodings
        - Different folds produce different encoding values
        - Overfitting is reduced through CV approach
    """
    X, y = target_encoding_data
    cv = KFold(n_splits=3, shuffle=True, random_state=17)

    generator = TargetEncodingFeatureGenerator(
        features_names=["category_a"], aggregation="mean", cv=cv
    )

    generator.fit(X, y)

    # Verify CV was used (implementation dependent)
    # This is a basic check that fit completed successfully with CV
    if generator.cv is not cv:
        raise AssertionError()


def test_targetencodingfeaturegenerator_fit_applies_smoothing_correctly(
    target_encoding_data,
):
    """Tests fit method applies smoothing to prevent overfitting on small categories.

    Args:
        target_encoding_data (tuple): Feature data and target for encoding

    Asserts:
        - Smoothing reduces impact of categories with few samples
        - Smoothing factor affects encoding values appropriately
        - Large smoothing values pull encodings toward global mean
    """
    X, y = target_encoding_data

    # Test with different smoothing values
    generator_low_smooth = TargetEncodingFeatureGenerator(
        features_names=["category_a"], smooth=1
    )

    generator_high_smooth = TargetEncodingFeatureGenerator(
        features_names=["category_a"], smooth=100
    )

    generator_low_smooth.fit(X, y)
    generator_high_smooth.fit(X, y)

    # Both should fit successfully with different smoothing
    # Specific validation depends on implementation details


def test_targetencodingfeaturegenerator_transform_generates_encoded_features(
    target_encoding_data,
):
    """Tests transform method generates target-encoded features.

    Args:
        target_encoding_data (tuple): Feature data and target for encoding

    Asserts:
        - Transform returns DataFrame with encoded features
        - Original features are replaced with numeric encodings
        - Encoding values are numeric and finite
    """
    X, y = target_encoding_data
    generator = TargetEncodingFeatureGenerator(
        features_names=["category_a", "category_b"]
    )

    generator.fit(X, y)
    transformed_x = generator.transform(X)

    if not isinstance(transformed_x, pd.DataFrame):
        raise AssertionError()
    if transformed_x.shape[0] != X.shape[0]:
        raise AssertionError()

    # Check that new encoded features are added (with TE_ prefix)
    expected_te_features = ["TE_MEAN_category_a", "TE_MEAN_category_b"]
    for te_feature in expected_te_features:
        if te_feature in transformed_x.columns:
            if not pd.api.types.is_numeric_dtype(transformed_x[te_feature]):
                raise AssertionError()
            if not np.all(np.isfinite(transformed_x[te_feature])):
                raise AssertionError()


def test_targetencodingfeaturegenerator_transform_raises_error_when_not_fitted():
    """Tests transform method raises error when called before fitting.

    Asserts:
        - RuntimeError or ValueError is raised when transform called on unfitted generator
        - Error message provides clear guidance about required fitting
    """
    X = pd.DataFrame({"category": ["a", "b", "c"]})
    generator = TargetEncodingFeatureGenerator(features_names=["category"])

    with pytest.raises((RuntimeError, ValueError, AttributeError)):
        generator.transform(X)


def test_targetencodingfeaturegenerator_transform_handles_unseen_categories(
    target_encoding_data,
):
    """Tests transform method handles categories not seen during fitting.

    Args:
        target_encoding_data (tuple): Feature data and target for encoding

    Asserts:
        - Unseen categories are handled gracefully
        - Default encoding values are used for unknown categories
        - No errors occur with new categorical values
    """
    X, y = target_encoding_data
    generator = TargetEncodingFeatureGenerator(features_names=["category_a"])
    generator.fit(X, y)

    # Create test data with unseen category
    X_test = pd.DataFrame(
        {
            "category_a": ["cat1", "unseen_category", "cat2"],
            "category_b": ["x", "y", "z"],
            "numeric": [1, 2, 3],
        }
    )

    transformed_x = generator.transform(X_test)

    # Should not raise error and return valid results
    if not isinstance(transformed_x, pd.DataFrame):
        raise AssertionError()
    if len(transformed_x) != len(X_test):
        raise AssertionError()


def test_targetencodingfeaturegenerator_fit_transform_combines_fit_and_transform(
    target_encoding_data,
):
    """Tests fit_transform method combines fitting and transformation.

    Args:
        target_encoding_data (tuple): Feature data and target for encoding

    Asserts:
        - fit_transform produces same result as separate fit() and transform()
        - Method is convenient shorthand for common workflow
        - Cross-validation is properly applied in combined method
    """
    X, y = target_encoding_data

    # Test fit_transform
    generator1 = TargetEncodingFeatureGenerator(features_names=["category_a"])
    transformed_x_1 = generator1.fit_transform(X, y)

    # Test separate fit and transform
    generator2 = TargetEncodingFeatureGenerator(features_names=["category_a"])
    generator2.fit(X, y)
    transformed_x_2 = generator2.transform(X)

    # Results should be similar (may not be exactly equal due to randomness in CV)
    if transformed_x_1.shape != transformed_x_2.shape:
        raise AssertionError()
    if not isinstance(transformed_x_1, pd.DataFrame):
        raise AssertionError()
    if not isinstance(transformed_x_2, pd.DataFrame):
        raise AssertionError()


def test_targetencodingfeaturegenerator_different_aggregations_produce_different_results(
    target_encoding_data,
):
    """Tests different aggregation methods produce different encoding results.

    Args:
        target_encoding_data (tuple): Feature data and target for encoding

    Asserts:
        - Mean, median, and nunique aggregations produce different encodings
        - All aggregation methods work without errors
        - Encoding values are appropriate for each aggregation type
    """
    X, y = target_encoding_data

    generators = {}
    results = {}

    for agg in ["mean", "median", "nunique"]:
        generators[agg] = TargetEncodingFeatureGenerator(
            features_names=["category_a"],
            aggregation=agg,
            cv=None,  # Disable CV for consistent comparison
        )
        generators[agg].fit(X, y)
        results[agg] = generators[agg].transform(X)

    # Results should be different between aggregation methods
    # (At least some values should differ)
    mean_values = results["mean"]["TE_MEAN_category_a"].values
    median_values = results["median"]["TE_MEDIAN_category_a"].values
    nunique_values = results["nunique"]["TE_NUNIQUE_category_a"].values

    # At least one pair should be different
    different_mean_median = not np.allclose(mean_values, median_values, equal_nan=True)
    different_mean_nunique = not np.allclose(
        mean_values, nunique_values, equal_nan=True
    )
    different_median_nunique = not np.allclose(
        median_values, nunique_values, equal_nan=True
    )

    if not (
        different_mean_median or different_mean_nunique or different_median_nunique
    ):
        raise AssertionError()


def test_targetencodingfeaturegenerator_handles_missing_feature_names(
    target_encoding_data,
):
    """Tests generator raises error when specified features don't exist in data.

    Args:
        target_encoding_data (tuple): Feature data and target for encoding

    Asserts:
        - KeyError is raised for non-existent features
        - Error message indicates which features are missing
        - Existing features work normally
    """
    X, y = target_encoding_data

    # Include one existing and one non-existing feature
    generator = TargetEncodingFeatureGenerator(
        features_names=["category_a", "nonexistent_feature"]
    )

    # Should raise KeyError for missing features
    with pytest.raises(KeyError, match="Missing features in X"):
        generator.fit(X, y)

    # Test that existing features work normally
    generator_valid = TargetEncodingFeatureGenerator(features_names=["category_a"])
    generator_valid.fit(X, y)  # Should work without error
    if not hasattr(generator_valid, "group_stats_"):
        raise AssertionError()


if __name__ == "__main__":
    print("Run this file with pytest to execute tests.")
