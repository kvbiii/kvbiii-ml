"""Tests for kvbiii_ml.data_processing.feature_engineering.categorical_aligner module."""

import numpy as np
import pandas as pd
import pytest

from kvbiii_ml.data_processing.feature_engineering.categorical_aligner import (
    CategoricalAligner,
)


@pytest.fixture
def categorical_training_data(test_settings) -> pd.DataFrame:
    """Provides training data with categorical features for testing.

    Args:
        test_settings: Test configuration fixture.

    Returns:
        pd.DataFrame: Training data with categorical columns.
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
def categorical_test_data() -> pd.DataFrame:
    """Provides test data with known and unknown categorical values.

    Returns:
        pd.DataFrame: Test data containing unseen categorical values.
    """
    return pd.DataFrame(
        {
            "color": ["blue", "purple", "yellow", "red"],
            "fuel": ["diesel", "hydrogen", "petrol", "electric"],
            "numeric": [15, 25, 35, 45],
        }
    )


def test_categoricalaligner_init_stores_default_parameters():
    """Tests CategoricalAligner initialization stores default constructor parameters.

    Asserts:
        - categorical_features defaults to None
        - fill_values defaults to None
        - warn_on_unknown defaults to True
        - categories_ and modes_ start as empty dicts
    """
    aligner = CategoricalAligner()
    if aligner.categorical_features is not None:
        raise AssertionError()
    if aligner.fill_values is not None:
        raise AssertionError()
    if aligner.warn_on_unknown is not True:
        raise AssertionError()
    if aligner.categories_ != {}:
        raise AssertionError()
    if aligner.modes_ != {}:
        raise AssertionError()


def test_categoricalaligner_init_stores_custom_parameters():
    """Tests CategoricalAligner initialization stores custom constructor parameters.

    Asserts:
        - categorical_features list is stored as provided
        - fill_values dict is stored as provided
        - warn_on_unknown flag is stored as provided
    """
    features = ["color", "fuel"]
    fill_values = {"color": "Unknown"}
    aligner = CategoricalAligner(
        categorical_features=features, fill_values=fill_values, warn_on_unknown=False
    )
    if aligner.categorical_features != features:
        raise AssertionError()
    if aligner.fill_values != fill_values:
        raise AssertionError()
    if aligner.warn_on_unknown is not False:
        raise AssertionError()


def test_categoricalaligner_create_fill_values_uses_default_value():
    """Tests create_fill_values classmethod applies the default fill value.

    Asserts:
        - Every feature listed maps to the default_value when no custom_fills given
        - Features not present in the DataFrame are excluded
    """
    df = pd.DataFrame({"color": ["red", "blue"], "fuel": ["petrol", "diesel"]})
    fill_values = CategoricalAligner.create_fill_values(
        df, categorical_features=["color", "fuel", "missing_feature"]
    )
    if fill_values != {"color": "mode", "fuel": "mode"}:
        raise AssertionError()


def test_categoricalaligner_create_fill_values_respects_custom_fills():
    """Tests create_fill_values classmethod overrides defaults with custom_fills.

    Asserts:
        - Feature present in custom_fills uses the custom value
        - Feature absent from custom_fills falls back to default_value
    """
    df = pd.DataFrame({"color": ["red", "blue"], "fuel": ["petrol", "diesel"]})
    fill_values = CategoricalAligner.create_fill_values(
        df,
        categorical_features=["color", "fuel"],
        custom_fills={"color": "Unknown"},
        default_value="Other",
    )
    if fill_values != {"color": "Unknown", "fuel": "Other"}:
        raise AssertionError()


def test_categoricalaligner_create_fill_values_handles_empty_inputs():
    """Tests create_fill_values classmethod with no categorical_features provided.

    Asserts:
        - An empty dict is returned when categorical_features is None
    """
    df = pd.DataFrame({"color": ["red", "blue"]})
    fill_values = CategoricalAligner.create_fill_values(df)
    if fill_values != {}:
        raise AssertionError()


def test_categoricalaligner_fit_auto_detects_categorical_features(
    categorical_training_data,
):
    """Tests fit auto-detects object/string/category columns when categorical_features is None.

    Args:
        categorical_training_data (pd.DataFrame): Training data with categoricals.

    Asserts:
        - categorical_features is populated with color and fuel only
        - numeric column is excluded from auto-detection
    """
    aligner = CategoricalAligner()
    aligner.fit(categorical_training_data)
    if set(aligner.categorical_features) != {"color", "fuel"}:
        raise AssertionError()
    if "numeric" in aligner.categorical_features:
        raise AssertionError()


def test_categoricalaligner_fit_computes_modes_and_categories(
    categorical_training_data,
):
    """Tests fit computes modes_ and categories_ from the training data.

    Args:
        categorical_training_data (pd.DataFrame): Training data with categoricals.

    Asserts:
        - modes_ contains the most frequent value for each categorical feature
        - categories_ contains the full observed category set for each feature
        - fit returns self
    """
    aligner = CategoricalAligner(categorical_features=["color", "fuel"])
    fitted_aligner = aligner.fit(categorical_training_data)
    if fitted_aligner is not aligner:
        raise AssertionError()
    if aligner.modes_["color"] not in {"red", "blue"}:
        raise AssertionError()
    if aligner.modes_["fuel"] not in {"petrol", "diesel"}:
        raise AssertionError()
    if set(aligner.categories_["color"]) != {"red", "blue", "green"}:
        raise AssertionError()
    if set(aligner.categories_["fuel"]) != {"petrol", "diesel", "electric"}:
        raise AssertionError()


def test_categoricalaligner_fit_categories_include_unseen_fill_value():
    """Tests fit includes the fill value in categories_ even when unseen in training data.

    Asserts:
        - The configured fill value is present in categories_ despite never
          appearing in the training data
    """
    df = pd.DataFrame({"color": pd.Categorical(["red", "blue", "red"])})
    aligner = CategoricalAligner(
        categorical_features=["color"], fill_values={"color": "Unknown"}
    )
    aligner.fit(df)
    if "Unknown" not in aligner.categories_["color"]:
        raise AssertionError()
    if set(aligner.categories_["color"]) != {"red", "blue", "Unknown"}:
        raise AssertionError()


def test_categoricalaligner_fit_skips_features_missing_from_dataframe():
    """Tests fit silently skips configured features absent from the input DataFrame.

    Asserts:
        - Missing feature does not appear in modes_ or categories_
        - Present feature is still processed normally
    """
    df = pd.DataFrame({"color": pd.Categorical(["red", "blue"])})
    aligner = CategoricalAligner(categorical_features=["color", "nonexistent"])
    aligner.fit(df)
    if "color" not in aligner.categories_:
        raise AssertionError()
    if "nonexistent" in aligner.categories_:
        raise AssertionError()
    if "nonexistent" in aligner.modes_:
        raise AssertionError()


def test_categoricalaligner_transform_fills_missing_values():
    """Tests transform fills NaN values with the fitted fill value.

    Asserts:
        - NaN entries are replaced with the feature's fill value
        - Resulting column is category dtype
    """
    train = pd.DataFrame({"color": pd.Categorical(["red", "blue", "red"])})
    test = pd.DataFrame({"color": ["red", np.nan, "blue"]})
    aligner = CategoricalAligner(categorical_features=["color"])
    aligner.fit(train)
    transformed = aligner.transform(test)
    if transformed.loc[1, "color"] != aligner.modes_["color"]:
        raise AssertionError()
    if not isinstance(transformed["color"].dtype, pd.CategoricalDtype):
        raise AssertionError()


def test_categoricalaligner_transform_replaces_unknown_categories(
    categorical_training_data, categorical_test_data
):
    """Tests transform replaces categories unseen during fit with the fill value.

    Args:
        categorical_training_data (pd.DataFrame): Training data.
        categorical_test_data (pd.DataFrame): Test data with unknown categories.

    Asserts:
        - Unknown color and fuel values are replaced with their fitted fill value
        - Known values are preserved unchanged
    """
    aligner = CategoricalAligner(categorical_features=["color", "fuel"])
    aligner.fit(categorical_training_data)
    transformed = aligner.transform(categorical_test_data)
    if transformed.loc[1, "color"] != aligner.modes_["color"]:
        raise AssertionError()
    if transformed.loc[1, "fuel"] != aligner.modes_["fuel"]:
        raise AssertionError()
    if transformed.loc[0, "color"] != "blue":
        raise AssertionError()
    if transformed.loc[3, "fuel"] != "electric":
        raise AssertionError()


def test_categoricalaligner_transform_warns_on_unknown_when_enabled(
    categorical_training_data, categorical_test_data
):
    """Tests transform raises a UserWarning when unknown categories are found and enabled.

    Args:
        categorical_training_data (pd.DataFrame): Training data.
        categorical_test_data (pd.DataFrame): Test data with unknown categories.

    Asserts:
        - A UserWarning is raised mentioning the unknown category count
    """
    aligner = CategoricalAligner(categorical_features=["color"], warn_on_unknown=True)
    aligner.fit(categorical_training_data)
    with pytest.warns(UserWarning, match="unknown categor"):
        aligner.transform(categorical_test_data)


def test_categoricalaligner_transform_does_not_warn_when_disabled(
    categorical_training_data, categorical_test_data, recwarn: pytest.WarningsRecorder
):
    """Tests transform does not raise a warning when warn_on_unknown is False.

    Args:
        categorical_training_data (pd.DataFrame): Training data.
        categorical_test_data (pd.DataFrame): Test data with unknown categories.
        recwarn (pytest.WarningsRecorder): Pytest fixture recording emitted warnings.

    Asserts:
        - No UserWarning is emitted during transform
    """
    aligner = CategoricalAligner(categorical_features=["color"], warn_on_unknown=False)
    aligner.fit(categorical_training_data)
    aligner.transform(categorical_test_data)
    user_warnings = [w for w in recwarn.list if issubclass(w.category, UserWarning)]
    if user_warnings:
        raise AssertionError(f"Unexpected warning(s) raised: {user_warnings}")


def test_categoricalaligner_transform_casts_to_category_with_full_category_set(
    categorical_training_data, categorical_test_data
):
    """Tests transform casts the output column to category dtype with the fitted set.

    Args:
        categorical_training_data (pd.DataFrame): Training data.
        categorical_test_data (pd.DataFrame): Test data.

    Asserts:
        - The transformed column's categories equal the fitted categories_ set
    """
    aligner = CategoricalAligner(categorical_features=["color"], warn_on_unknown=False)
    aligner.fit(categorical_training_data)
    transformed = aligner.transform(categorical_test_data)
    if set(transformed["color"].cat.categories) != set(aligner.categories_["color"]):
        raise AssertionError()


def test_categoricalaligner_transform_before_fit_is_noop_copy():
    """Tests transform before fit is a silent no-op copy since categories_ starts as {}.

    Asserts:
        - Transformed output equals the input DataFrame
        - Transformed output is a distinct object from the input
    """
    aligner = CategoricalAligner()
    df = pd.DataFrame({"color": ["red", "blue"]})
    transformed = aligner.transform(df)
    pd.testing.assert_frame_equal(transformed, df)
    if transformed is df:
        raise AssertionError()


def test_categoricalaligner_transform_skips_features_missing_from_input():
    """Tests transform skips fitted features that are absent from the input DataFrame.

    Asserts:
        - Transform completes without error when a fitted feature is missing
        - Remaining columns are returned unchanged
    """
    train = pd.DataFrame(
        {"color": pd.Categorical(["red", "blue"]), "fuel": pd.Categorical(["a", "b"])}
    )
    aligner = CategoricalAligner(categorical_features=["color", "fuel"])
    aligner.fit(train)
    test = pd.DataFrame({"color": ["red", "blue"]})
    transformed = aligner.transform(test)
    if "fuel" in transformed.columns:
        raise AssertionError()
    if not isinstance(transformed["color"].dtype, pd.CategoricalDtype):
        raise AssertionError()


def test_categoricalaligner_does_not_expose_get_feature_names_out():
    """Tests that CategoricalAligner does not define get_feature_names_out.

    The real source class only defines create_fill_values, fit, and transform;
    it neither implements nor inherits a get_feature_names_out method from
    sklearn's BaseEstimator/TransformerMixin in the installed sklearn version.

    Asserts:
        - The instance has no get_feature_names_out attribute
    """
    aligner = CategoricalAligner()
    if hasattr(aligner, "get_feature_names_out"):
        raise AssertionError()


if __name__ == "__main__":
    print("Run this file with pytest to execute tests.")
