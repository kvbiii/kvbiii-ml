"""Tests for CrossFeatureGenerator class in cross_encoding module."""

import numpy as np
import pandas as pd
import pytest

from kvbiii_ml.data_processing.feature_engineering.cross_encoding import (
    CrossFeatureGenerator,
)


class TestCrossFeatureGenerator:
    """Test suite for CrossFeatureGenerator class."""

    def test_crossfeaturegenerator_init_default_parameters(self):
        """Tests CrossFeatureGenerator initialization with default parameters.

        Asserts:
            - Default parameters are set correctly
            - Empty feature_names list is handled
            - Default degree is 2
            - Default separator is '_'
        """
        generator = CrossFeatureGenerator()

        assert generator.features_names == []
        assert generator.degree == 2
        assert generator.separator == "_"
        assert generator.feature_combinations_ == []
        assert generator.encoding_maps_ == {}

    def test_crossfeaturegenerator_init_custom_parameters(self):
        """Tests CrossFeatureGenerator initialization with custom parameters.

        Asserts:
            - Custom parameters are set correctly
            - Feature names list is preserved
            - Custom degree and separator are applied
        """
        features = ["col1", "col2", "col3"]
        generator = CrossFeatureGenerator(
            features_names=features, degree=3, separator="-"
        )

        assert generator.features_names == features
        assert generator.degree == 3
        assert generator.separator == "-"

    def test_crossfeaturegenerator_validate_init_params_invalid_features(self):
        """Tests CrossFeatureGenerator parameter validation with invalid feature names.

        Asserts:
            - ValueError is raised for non-list feature names
            - ValueError is raised for non-string feature names
        """
        with pytest.raises(
            ValueError, match="features_names must be a list of strings"
        ):
            CrossFeatureGenerator(features_names="not_a_list")

        with pytest.raises(
            ValueError, match="features_names must be a list of strings"
        ):
            CrossFeatureGenerator(features_names=[1, 2, 3])

    def test_crossfeaturegenerator_validate_init_params_invalid_degree(self):
        """Tests CrossFeatureGenerator parameter validation with invalid degree.

        Asserts:
            - ValueError is raised for non-integer degree
            - ValueError is raised for degree less than 2
        """
        with pytest.raises(ValueError, match="degree must be an integer >= 2"):
            CrossFeatureGenerator(degree=1.5)

        with pytest.raises(ValueError, match="degree must be an integer >= 2"):
            CrossFeatureGenerator(degree=1)

    def test_crossfeaturegenerator_validate_init_params_invalid_separator(self):
        """Tests CrossFeatureGenerator parameter validation with invalid separator.

        Asserts:
            - ValueError is raised for non-string separator
        """
        with pytest.raises(ValueError, match="separator must be a string"):
            CrossFeatureGenerator(separator=123)

    def test_crossfeaturegenerator_validate_init_params_insufficient_features(self):
        """Tests CrossFeatureGenerator parameter validation with insufficient features.

        Asserts:
            - ValueError is raised when feature count is less than degree
        """
        with pytest.raises(ValueError, match="Number of features .* must be >= degree"):
            CrossFeatureGenerator(features_names=["col1"], degree=2)

    def test_crossfeaturegenerator_is_numerical_combo_all_numeric(
        self, sample_dataframe
    ):
        """Tests _is_numerical_combo method with all numeric columns.

        Args:
            sample_dataframe (pd.DataFrame): Test DataFrame fixture

        Asserts:
            - Returns True for all numeric columns
            - Correctly identifies numeric data types
        """
        generator = CrossFeatureGenerator()

        # Create dataframe with numeric columns
        df = pd.DataFrame(
            {"num1": [1, 2, 3], "num2": [1.5, 2.5, 3.5], "cat1": ["A", "B", "C"]}
        )

        assert generator._is_numerical_combo(df, ["num1", "num2"]) == True

    def test_crossfeaturegenerator_is_numerical_combo_mixed_types(self):
        """Tests _is_numerical_combo method with mixed data types.

        Asserts:
            - Returns False for mixed numeric and categorical columns
            - Correctly identifies mixed data types
        """
        generator = CrossFeatureGenerator()

        df = pd.DataFrame({"num1": [1, 2, 3], "cat1": ["A", "B", "C"]})

        assert generator._is_numerical_combo(df, ["num1", "cat1"]) == False

    def test_crossfeaturegenerator_fit_categorical_features(self):
        """Tests fit method with categorical features.

        Asserts:
            - Encoding maps are created for categorical combinations
            - Feature combinations are generated correctly
            - Combined value maps are created
        """
        df = pd.DataFrame({"A": ["a", "b", "a", "c"], "B": ["x", "y", "x", "z"]})

        generator = CrossFeatureGenerator(features_names=["A", "B"], degree=2)
        fitted_generator = generator.fit(df)

        assert fitted_generator is generator
        assert len(generator.feature_combinations_) == 1
        assert ("A", "B") in generator.feature_combinations_
        assert "A-B" in generator.encoding_maps_
        assert (
            len(generator.encoding_maps_["A-B"]) == 3
        )  # a_x, b_y, c_z -> unique combinations

    def test_crossfeaturegenerator_fit_numerical_features(self):
        """Tests fit method with numerical features.

        Asserts:
            - Numerical combinations are tracked separately
            - No encoding maps created for numerical features
            - Combined value maps contain multiplied values
        """
        df = pd.DataFrame({"D": [10, 20, 10, 30], "E": [0.1, 0.2, 0.1, 0.3]})

        generator = CrossFeatureGenerator(features_names=["D", "E"], degree=2)
        generator.fit(df)

        assert "D-E" in generator.numerical_combos_
        assert "D-E" not in generator.encoding_maps_
        assert "D-E" in generator.combined_value_maps_

    def test_crossfeaturegenerator_fit_empty_features_uses_all_columns(self):
        """Tests fit method with empty features_names uses all DataFrame columns.

        Asserts:
            - All DataFrame columns are used when features_names is empty
            - Feature combinations are generated from all columns
        """
        df = pd.DataFrame({"A": ["a", "b"], "B": ["x", "y"], "C": ["1", "2"]})

        generator = CrossFeatureGenerator(degree=2)
        generator.fit(df)

        assert set(generator.features_names) == {"A", "B", "C"}
        assert len(generator.feature_combinations_) == 3  # AB, AC, BC

    def test_crossfeaturegenerator_fit_insufficient_columns_raises_error(self):
        """Tests fit method with insufficient columns raises ValueError.

        Asserts:
            - ValueError is raised when DataFrame has fewer columns than degree
        """
        df = pd.DataFrame({"A": ["a", "b"]})

        generator = CrossFeatureGenerator(degree=2)

        with pytest.raises(ValueError, match="Number of features .* must be >= degree"):
            generator.fit(df)

    def test_crossfeaturegenerator_transform_categorical_features(self):
        """Tests transform method with categorical features.

        Asserts:
            - New encoded columns are added to DataFrame
            - Original columns are preserved
            - Encoded values are integers
        """
        df = pd.DataFrame({"A": ["a", "b", "a"], "B": ["x", "y", "x"]})

        generator = CrossFeatureGenerator(features_names=["A", "B"], degree=2)
        generator.fit(df)
        result = generator.transform(df)

        assert "A-B" in result.columns
        assert "A" in result.columns
        assert "B" in result.columns
        assert result["A-B"].dtype == int
        assert len(result) == len(df)

    def test_crossfeaturegenerator_transform_numerical_features(self):
        """Tests transform method with numerical features.

        Asserts:
            - New product columns are added to DataFrame
            - Values are products of original columns
            - Original columns are preserved
        """
        df = pd.DataFrame({"D": [2, 3, 4], "E": [5, 6, 7]})

        generator = CrossFeatureGenerator(features_names=["D", "E"], degree=2)
        generator.fit(df)
        result = generator.transform(df)

        assert "D-E" in result.columns
        assert result["D-E"].tolist() == [10, 18, 28]  # 2*5, 3*6, 4*7

    def test_crossfeaturegenerator_transform_unfitted_raises_error(self):
        """Tests transform method raises error when not fitted.

        Asserts:
            - ValueError is raised when transform is called before fit
        """
        df = pd.DataFrame({"A": ["a", "b"], "B": ["x", "y"]})
        generator = CrossFeatureGenerator()

        with pytest.raises(
            ValueError, match="CrossFeatureGenerator must be fitted before transform"
        ):
            generator.transform(df)

    def test_crossfeaturegenerator_transform_unseen_values_handled(self):
        """Tests transform method handles unseen categorical values.

        Asserts:
            - Unseen values are encoded as -1
            - Transform completes without errors
        """
        train_df = pd.DataFrame({"A": ["a", "b"], "B": ["x", "y"]})

        test_df = pd.DataFrame(
            {"A": ["c", "a"], "B": ["z", "x"]}
        )  # 'c' is unseen  # 'z' is unseen

        generator = CrossFeatureGenerator(features_names=["A", "B"], degree=2)
        generator.fit(train_df)
        result = generator.transform(test_df)

        assert "A-B" in result.columns
        assert -1 in result["A-B"].values  # Unseen combination should be -1

    def test_crossfeaturegenerator_fit_transform_equivalent_to_fit_then_transform(self):
        """Tests fit_transform produces same result as fit then transform.

        Asserts:
            - fit_transform result equals separate fit and transform
            - Both methods produce identical DataFrames
        """
        df = pd.DataFrame({"A": ["a", "b", "a"], "B": ["x", "y", "x"]})

        generator1 = CrossFeatureGenerator(features_names=["A", "B"], degree=2)
        result1 = generator1.fit_transform(df)

        generator2 = CrossFeatureGenerator(features_names=["A", "B"], degree=2)
        generator2.fit(df)
        result2 = generator2.transform(df)

        pd.testing.assert_frame_equal(result1, result2)

    def test_crossfeaturegenerator_get_feature_names_empty_before_fit(self):
        """Tests get_feature_names returns empty list before fitting.

        Asserts:
            - Empty list is returned when no features are generated
        """
        generator = CrossFeatureGenerator()

        assert generator.get_feature_names() == []

    def test_crossfeaturegenerator_get_feature_names_after_fit(self):
        """Tests get_feature_names returns correct names after fitting.

        Asserts:
            - Correct feature names are returned after fitting
            - Names follow the expected format
        """
        df = pd.DataFrame({"A": ["a", "b"], "B": ["x", "y"], "C": ["1", "2"]})

        generator = CrossFeatureGenerator(features_names=["A", "B", "C"], degree=2)
        generator.fit(df)
        feature_names = generator.get_feature_names()

        expected_names = ["A-B", "A-C", "B-C"]
        assert sorted(feature_names) == sorted(expected_names)

    def test_crossfeaturegenerator_custom_separator(self):
        """Tests CrossFeatureGenerator with custom separator.

        Asserts:
            - Custom separator is used in feature value combination
            - Feature names still use hyphen separator
        """
        df = pd.DataFrame({"A": ["a", "b"], "B": ["x", "y"]})

        generator = CrossFeatureGenerator(features_names=["A", "B"], separator="|")
        generator.fit(df)

        # Check that the combined values use the custom separator
        combined_values = generator.combined_value_maps_["A-B"]
        assert all("|" in str(val) for val in combined_values.unique())

    def test_crossfeaturegenerator_degree_three_combinations(self):
        """Tests CrossFeatureGenerator with degree 3 combinations.

        Asserts:
            - Correct number of 3-way combinations are generated
            - All combinations include exactly 3 features
        """
        df = pd.DataFrame(
            {"A": ["a", "b"], "B": ["x", "y"], "C": ["1", "2"], "D": ["p", "q"]}
        )

        generator = CrossFeatureGenerator(features_names=["A", "B", "C", "D"], degree=3)
        generator.fit(df)

        # 4 choose 3 = 4 combinations
        assert len(generator.feature_combinations_) == 4
        assert all(len(combo) == 3 for combo in generator.feature_combinations_)


if __name__ == "__main__":
    print("Run this file with pytest to execute tests.")
