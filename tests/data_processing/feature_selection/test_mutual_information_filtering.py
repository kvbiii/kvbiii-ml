"""Tests for kvbiii_ml.data_processing.feature_selection.mutual_information_filtering module."""

from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from kvbiii_ml.data_processing.feature_selection.mutual_information_filtering import (
    MutualInformationFiltering,
)


@pytest.fixture
def classification_feature_data(test_settings):
    """Provides feature data with target for classification testing.

    Args:
        test_settings: Test configuration fixture

    Returns:
        tuple[pd.DataFrame, pd.Series]: Feature matrix and binary classification target
    """
    np.random.seed(test_settings.SEED)
    X = pd.DataFrame(
        {
            "informative_1": np.random.randn(test_settings.N_SAMPLES),
            "informative_2": np.random.randn(test_settings.N_SAMPLES),
            "noise_1": np.random.randn(test_settings.N_SAMPLES),
            "noise_2": np.random.randn(test_settings.N_SAMPLES),
            "categorical": pd.Categorical(
                np.random.choice(["A", "B", "C"], size=test_settings.N_SAMPLES)
            ),
        }
    )

    # Create target that depends on informative features
    y = pd.Series(
        (
            (
                X["informative_1"]
                + X["informative_2"]
                + np.random.randn(test_settings.N_SAMPLES) * 0.1
            )
            > 0
        ).astype(int),
        name="target",
    )

    return X, y


@pytest.fixture
def regression_feature_data(test_settings):
    """Provides feature data with continuous target for regression testing.

    Args:
        test_settings: Test configuration fixture

    Returns:
        tuple[pd.DataFrame, pd.Series]: Feature matrix and continuous regression target
    """
    np.random.seed(test_settings.SEED)
    X = pd.DataFrame(
        {
            "predictive_1": np.random.randn(test_settings.N_SAMPLES),
            "predictive_2": np.random.randn(test_settings.N_SAMPLES),
            "irrelevant_1": np.random.randn(test_settings.N_SAMPLES),
            "irrelevant_2": np.random.randn(test_settings.N_SAMPLES),
        }
    )

    # Create target that depends on predictive features
    y = pd.Series(
        2 * X["predictive_1"]
        + X["predictive_2"]
        + np.random.randn(test_settings.N_SAMPLES) * 0.1,
        name="target",
    )

    return X, y


def test_mutualinformationfiltering_init_creates_classification_instance():
    """Tests MutualInformationFiltering initialization for classification problems.

    Asserts:
        - Instance is created with classification problem type
        - Default parameters are set correctly
        - MI computation parameters are configured appropriately
    """
    mif = MutualInformationFiltering(problem_type="classification")

    if mif.problem_type != "classification":
        raise AssertionError()
    if mif.threshold != 0.1:
        raise AssertionError()
    if mif.keep_top_k is not None:
        raise AssertionError()
    if mif.verbose is not False:
        raise AssertionError()
    if mif.selected_features_ != []:
        raise AssertionError()
    if mif.threshold_border_ is not None:
        raise AssertionError()

    # Check MI computation parameters
    if "n_neighbors" not in mif.mi_kwargs:
        raise AssertionError()
    if "random_state" not in mif.mi_kwargs:
        raise AssertionError()
    if mif.mi_kwargs["random_state"] != 17:
        raise AssertionError()


def test_mutualinformationfiltering_init_creates_regression_instance():
    """Tests MutualInformationFiltering initialization for regression problems.

    Asserts:
        - Instance is created with regression problem type
        - Custom parameters can be set during initialization
        - Instance state is properly initialized
    """
    mif = MutualInformationFiltering(
        problem_type="regression", threshold=0.05, keep_top_k=5, verbose=True
    )

    if mif.problem_type != "regression":
        raise AssertionError()
    if mif.threshold != 0.05:
        raise AssertionError()
    if mif.keep_top_k != 5:
        raise AssertionError()
    if mif.verbose is not True:
        raise AssertionError()


def test_mutualinformationfiltering_init_raises_error_for_invalid_problem_type():
    """Tests MutualInformationFiltering initialization rejects invalid problem types.

    Asserts:
        - ValueError is raised for unsupported problem types
        - Error message provides clear guidance about valid options
    """
    with pytest.raises(
        ValueError, match="problem_type must be either 'classification' or 'regression'"
    ):
        MutualInformationFiltering(problem_type="invalid_type")


def test_mutualinformationfiltering_fit_computes_mi_scores_for_classification(
    classification_feature_data,
):
    """Tests fit method computes mutual information scores for classification.

    Args:
        classification_feature_data (tuple): Feature data and classification target

    Asserts:
        - Fit method completes without errors
        - Selected features are computed based on threshold
        - Instance state is updated with fitting results
    """
    X, y = classification_feature_data
    mif = MutualInformationFiltering(
        problem_type="classification", threshold=0.0
    )  # Keep all features

    fitted_mif = mif.fit(X, y)

    if fitted_mif is not mif:
        raise AssertionError()
    if not len(mif.selected_features_) > 0:
        raise AssertionError()
    if not all(feature in X.columns for feature in mif.selected_features_):
        raise AssertionError()
    if mif.threshold_border_ != 0.0:
        raise AssertionError()


def test_mutualinformationfiltering_fit_computes_mi_scores_for_regression(
    regression_feature_data,
):
    """Tests fit method computes mutual information scores for regression.

    Args:
        regression_feature_data (tuple): Feature data and regression target

    Asserts:
        - Regression MI computation works correctly
        - Features are selected based on continuous target relationships
        - Threshold filtering works for regression scenarios
    """
    X, y = regression_feature_data
    mif = MutualInformationFiltering(problem_type="regression", threshold=0.0)

    mif.fit(X, y)

    if not len(mif.selected_features_) > 0:
        raise AssertionError()
    if not isinstance(mif.threshold_border_, float):
        raise AssertionError()
    if not all(isinstance(feature, str) for feature in mif.selected_features_):
        raise AssertionError()


def test_mutualinformationfiltering_fit_handles_top_k_selection(
    classification_feature_data,
):
    """Tests fit method handles top-k feature selection correctly.

    Args:
        classification_feature_data (tuple): Feature data and classification target

    Asserts:
        - Exactly k features are selected when keep_top_k is specified
        - Threshold border is set to k-th highest MI score
        - Selected features are the top-k by mutual information
    """
    X, y = classification_feature_data
    k = 3
    mif = MutualInformationFiltering(problem_type="classification", keep_top_k=k)

    mif.fit(X, y)

    if len(mif.selected_features_) != k:
        raise AssertionError()
    if mif.threshold_border_ is None:
        raise AssertionError()
    if not isinstance(mif.threshold_border_, float):
        raise AssertionError()


def test_mutualinformationfiltering_fit_handles_top_k_larger_than_features(
    classification_feature_data,
):
    """Tests fit method handles keep_top_k larger than available features.

    Args:
        classification_feature_data (tuple): Feature data and classification target

    Asserts:
        - All features are selected when k exceeds number of features
        - No errors are raised for oversized k values
        - Selection behaves gracefully with boundary conditions
    """
    X, y = classification_feature_data
    k = X.shape[1] + 10  # More than available features
    mif = MutualInformationFiltering(problem_type="classification", keep_top_k=k)

    mif.fit(X, y)

    if len(mif.selected_features_) != X.shape[1]:
        raise AssertionError()
    if set(mif.selected_features_) != set(X.columns):
        raise AssertionError()


def test_mutualinformationfiltering_fit_raises_error_for_invalid_top_k():
    """Tests fit method raises error for invalid keep_top_k values.

    Asserts:
        - ValueError is raised for non-positive keep_top_k values
        - Error message explains valid range for k parameter
    """
    X = pd.DataFrame({"feature": [1, 2, 3]})
    y = pd.Series([0, 1, 0])

    mif = MutualInformationFiltering(problem_type="classification", keep_top_k=0)

    with pytest.raises(ValueError, match="keep_top_k must be a positive integer"):
        mif.fit(X, y)


def test_mutualinformationfiltering_transform_returns_selected_features(
    classification_feature_data,
):
    """Tests transform method returns only selected features.

    Args:
        classification_feature_data (tuple): Feature data and classification target

    Asserts:
        - Transform returns DataFrame with only selected features
        - Feature order and data integrity are preserved
        - Output shape matches number of selected features
    """
    X, y = classification_feature_data
    mif = MutualInformationFiltering(problem_type="classification", keep_top_k=3)
    mif.fit(X, y)

    transformed_x = mif.transform(X)

    if not isinstance(transformed_x, pd.DataFrame):
        raise AssertionError()
    if transformed_x.shape[1] != 3:
        raise AssertionError()
    if transformed_x.shape[0] != X.shape[0]:
        raise AssertionError()
    if list(transformed_x.columns) != mif.selected_features_:
        raise AssertionError()


def test_mutualinformationfiltering_transform_raises_error_when_not_fitted():
    """Tests transform method raises error when called before fitting.

    Asserts:
        - ValueError is raised when transform called on unfitted instance
        - Error message provides clear guidance about required fitting
    """
    X = pd.DataFrame({"feature": [1, 2, 3]})
    mif = MutualInformationFiltering(problem_type="classification")

    with pytest.raises(
        ValueError, match="The filter has not been fitted yet. Call fit\\(\\) first"
    ):
        mif.transform(X)


def test_mutualinformationfiltering_fit_transform_combines_fit_and_transform(
    classification_feature_data,
):
    """Tests fit_transform method combines fitting and transformation in one call.

    Args:
        classification_feature_data (tuple): Feature data and classification target

    Asserts:
        - fit_transform produces same result as separate fit() and transform() calls
        - Method is convenient shorthand for common workflow
        - Instance state is properly updated during fit_transform
    """
    X, y = classification_feature_data

    # Test fit_transform
    mif1 = MutualInformationFiltering(problem_type="classification", keep_top_k=2)
    transformed_x_1 = mif1.fit_transform(X, y)

    # Test separate fit and transform
    mif2 = MutualInformationFiltering(problem_type="classification", keep_top_k=2)
    mif2.fit(X, y)
    transformed_x_2 = mif2.transform(X)

    pd.testing.assert_frame_equal(transformed_x_1, transformed_x_2)
    if mif1.selected_features_ != mif2.selected_features_:
        raise AssertionError()


def test_mutualinformationfiltering_prepare_x_y_for_mi_validates_inputs():
    """Tests _prepare_X_y_for_mi method validates input data types correctly.

    Asserts:
        - ValueError is raised for non-DataFrame X inputs
        - ValueError is raised for invalid y input types
        - Error messages provide clear guidance about expected types
    """
    mif = MutualInformationFiltering(problem_type="classification")

    # Test invalid X type
    with pytest.raises(ValueError, match="X must be a pandas DataFrame"):
        mif._prepare_X_y_for_mi(np.array([[1, 2], [3, 4]]), pd.Series([0, 1]))

    # Test invalid y type
    with pytest.raises(ValueError, match="y must be a pandas Series or numpy array"):
        mif._prepare_X_y_for_mi(pd.DataFrame({"A": [1, 2]}), [[0, 1]])


def test_mutualinformationfiltering_prepare_x_y_for_mi_handles_categorical_features():
    """Tests _prepare_X_y_for_mi method properly encodes categorical features.

    Asserts:
        - Categorical features are converted to numeric codes
        - discrete_features parameter is set appropriately for MI computation
        - Non-categorical features remain unchanged
    """
    X = pd.DataFrame(
        {"numeric": [1.0, 2.0, 3.0], "categorical": pd.Categorical(["A", "B", "A"])}
    )
    y = pd.Series([0, 1, 0])

    mif = MutualInformationFiltering(problem_type="classification")
    prepared_x, _prepared_y = mif._prepare_X_y_for_mi(X, y)

    # Categorical should be converted to codes
    if not pd.api.types.is_numeric_dtype(prepared_x["categorical"]):
        raise AssertionError()

    # discrete_features should be set for MI computation
    if "discrete_features" not in mif.mi_kwargs:
        raise AssertionError()
    if mif.mi_kwargs["discrete_features"][1] != True:
        raise AssertionError()
    if mif.mi_kwargs["discrete_features"][0] != False:
        raise AssertionError()


def test_mutualinformationfiltering_compute_mi_scores_handles_nan_values(
    classification_feature_data,
):
    """Tests _compute_mi_scores method handles NaN MI scores appropriately.

    Args:
        classification_feature_data (tuple): Feature data and classification target

    Asserts:
        - NaN MI scores are converted to 0.0
        - All MI scores are finite numeric values
        - Method returns valid score dictionary
    """
    X, y = classification_feature_data
    mif = MutualInformationFiltering(problem_type="classification")

    # Mock MI computation to return NaN values
    with patch(
        "kvbiii_ml.data_processing.eda.data_analysis.mutual_info_classif"
    ) as mock_mi:
        mock_mi.return_value = np.array([0.5, np.nan, 0.3, np.inf, 0.1])

        scores = mif._compute_mi_scores(X, y)

        if not all(np.isfinite(score) for score in scores.values()):
            raise AssertionError()
        if not all(score >= 0.0 for score in scores.values()):
            raise AssertionError()


def test_mutualinformationfiltering_verbose_mode_prints_selection_info(
    classification_feature_data, capsys
):
    """Tests verbose mode prints informative messages during feature selection.

    Args:
        classification_feature_data (tuple): Feature data and classification target
        capsys: Pytest fixture for capturing stdout

    Asserts:
        - Verbose mode prints threshold information
        - Messages contain relevant selection statistics
        - Output format is user-friendly and informative
    """
    X, y = classification_feature_data

    # Test threshold-based selection
    mif_threshold = MutualInformationFiltering(
        problem_type="classification", threshold=0.01, verbose=True
    )
    mif_threshold.fit(X, y)

    captured = capsys.readouterr()
    if "nominal MI threshold" not in captured.out:
        raise AssertionError()
    if "Selected" not in captured.out:
        raise AssertionError()

    # Test top-k selection
    mif_topk = MutualInformationFiltering(
        problem_type="classification", keep_top_k=2, verbose=True
    )
    mif_topk.fit(X, y)

    captured = capsys.readouterr()
    if "top-k selection" not in captured.out:
        raise AssertionError()
    if "Implied MI threshold" not in captured.out:
        raise AssertionError()


def test_mutualinformationfiltering_handles_empty_feature_selection(
    classification_feature_data,
):
    """Tests MutualInformationFiltering behavior when no features meet threshold.

    Args:
        classification_feature_data (tuple): Feature data and classification target

    Asserts:
        - Empty feature selection results in error on transform
        - fit() completes without error even with empty selection
        - Instance state remains consistent with empty selection
    """
    X, y = classification_feature_data
    mif = MutualInformationFiltering(
        problem_type="classification", threshold=999.0
    )  # Very high threshold
    mif.fit(X, y)

    if len(mif.selected_features_) != 0:
        raise AssertionError()

    # Transform should raise error since implementation treats empty selection as not fitted
    with pytest.raises(ValueError, match="The filter has not been fitted yet"):
        mif.transform(X)


if __name__ == "__main__":
    print("Run this file with pytest to execute tests.")
