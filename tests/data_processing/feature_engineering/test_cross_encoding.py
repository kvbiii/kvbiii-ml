"""Tests for CrossFeatureGenerator in kvbiii_ml.data_processing.feature_engineering.cross_encoding."""

import numpy as np
import pandas as pd
import pytest

from kvbiii_ml.data_processing.feature_engineering.cross_encoding import (
    CrossFeatureGenerator,
)


@pytest.fixture
def mixed_dataframe() -> pd.DataFrame:
    """Provides a small DataFrame with categorical and numeric columns.

    Returns:
        pd.DataFrame: DataFrame with columns A, B (categorical), C, D (numeric).
    """
    return pd.DataFrame(
        {
            "A": ["a", "b", "a", "c"],
            "B": ["x", "y", "x", "z"],
            "C": [1, 2, 1, 3],
            "D": [10, 20, 10, 30],
        }
    )


def test_crossfeaturegenerator_init_default_parameters():
    """Tests CrossFeatureGenerator initialization with default parameters.

    Asserts:
        - Empty features_names list is the default
        - Default degree is 2
        - Default separator is "_"
        - feature_combinations_, encoding_maps_, and numerical_combos_ start empty
    """
    generator = CrossFeatureGenerator()
    if generator.features_names != []:
        raise AssertionError()
    if generator.degree != 2:
        raise AssertionError()
    if generator.separator != "_":
        raise AssertionError()
    if generator.feature_combinations_ != []:
        raise AssertionError()
    if generator.encoding_maps_ != {}:
        raise AssertionError()
    if generator.numerical_combos_ != set():
        raise AssertionError()
    if generator.batch_size != 10:
        raise AssertionError()
    if generator.chunk_size != 50000:
        raise AssertionError()


def test_crossfeaturegenerator_init_custom_parameters():
    """Tests CrossFeatureGenerator initialization with custom parameters.

    Asserts:
        - Custom features_names, degree, and separator are stored as provided
    """
    features = ["col1", "col2", "col3"]
    generator = CrossFeatureGenerator(features_names=features, degree=3, separator="-")
    if generator.features_names != features:
        raise AssertionError()
    if generator.degree != 3:
        raise AssertionError()
    if generator.separator != "-":
        raise AssertionError()


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"features_names": "not_a_list"}, "features_names must be a list of strings"),
        ({"features_names": [1, 2, 3]}, "features_names must be a list of strings"),
        ({"degree": 1.5}, "degree must be an integer >= 2"),
        ({"degree": 1}, "degree must be an integer >= 2"),
        ({"separator": 123}, "separator must be a string"),
        ({"batch_size": 0}, "batch_size must be a positive integer"),
        ({"batch_size": -3}, "batch_size must be a positive integer"),
        ({"chunk_size": 0}, "chunk_size must be a positive integer"),
        ({"chunk_size": -1}, "chunk_size must be a positive integer"),
        (
            {"features_names": ["A"], "degree": 2},
            "Number of features .* must be >= degree",
        ),
    ],
)
def test_crossfeaturegenerator_validate_init_params_raises(kwargs: dict, match: str):
    """Tests that invalid constructor parameters raise ValueError with a matching message.

    Args:
        kwargs (dict): Constructor keyword arguments under test.
        match (str): Regex expected to match the raised error message.

    Asserts:
        - ValueError is raised with the expected message for each invalid input
    """
    with pytest.raises(ValueError, match=match):
        CrossFeatureGenerator(**kwargs)


def test_crossfeaturegenerator_legacy_batch_chunk_kwargs_are_applied():
    """Tests that legacy batch_size/chunk_size kwargs populate the processing config.

    Asserts:
        - batch_size and chunk_size properties reflect the passed kwargs
    """
    generator = CrossFeatureGenerator(batch_size=3, chunk_size=100)
    if generator.batch_size != 3:
        raise AssertionError()
    if generator.chunk_size != 100:
        raise AssertionError()


def test_crossfeaturegenerator_processing_dict_is_applied():
    """Tests that a processing dict configures batch_size and chunk_size.

    Asserts:
        - batch_size and chunk_size properties reflect the processing dict values
    """
    generator = CrossFeatureGenerator(processing={"batch_size": 7, "chunk_size": 250})
    if generator.batch_size != 7:
        raise AssertionError()
    if generator.chunk_size != 250:
        raise AssertionError()


def test_crossfeaturegenerator_unexpected_kwarg_raises_value_error():
    """Tests that an unrecognized keyword argument raises ValueError.

    Asserts:
        - ValueError is raised mentioning the unexpected argument name
    """
    with pytest.raises(ValueError, match="Unexpected arguments: unknown_kwarg"):
        CrossFeatureGenerator(unknown_kwarg=5)


def test_crossfeaturegenerator_fit_empty_features_uses_all_columns(mixed_dataframe):
    """Tests fit defaults features_names to all DataFrame columns when left empty.

    Args:
        mixed_dataframe (pd.DataFrame): Fixture DataFrame with mixed dtypes.

    Asserts:
        - features_names is populated with every column of the input DataFrame
        - Feature combinations are generated from all columns
    """
    generator = CrossFeatureGenerator(degree=2)
    generator.fit(mixed_dataframe)
    if set(generator.features_names) != set(mixed_dataframe.columns):
        raise AssertionError()
    if len(generator.feature_combinations_) != 6:
        raise AssertionError()


def test_crossfeaturegenerator_fit_insufficient_columns_raises_error():
    """Tests fit raises ValueError when the DataFrame has fewer columns than degree.

    Asserts:
        - ValueError is raised mentioning the required minimum feature count
    """
    df = pd.DataFrame({"A": ["a", "b"]})
    generator = CrossFeatureGenerator(degree=2)
    with pytest.raises(ValueError, match="Number of features .* must be >= degree"):
        generator.fit(df)


def test_crossfeaturegenerator_fit_categorical_combo_uses_default_separator():
    """Tests that fit builds combo column names via the default separator "_".

    Asserts:
        - Combo name "A_B" is present in encoding_maps_ for the default separator
        - encoding_maps_ contains one integer code per unique combined value
    """
    df = pd.DataFrame({"A": ["a", "b", "a", "c"], "B": ["x", "y", "x", "z"]})
    generator = CrossFeatureGenerator(features_names=["A", "B"], degree=2)
    fitted_generator = generator.fit(df)
    if fitted_generator is not generator:
        raise AssertionError()
    if ("A", "B") not in generator.feature_combinations_:
        raise AssertionError()
    if "A_B" not in generator.encoding_maps_:
        raise AssertionError()
    if generator.encoding_maps_["A_B"] != {"a_x": 0, "b_y": 1, "c_z": 2}:
        raise AssertionError()


def test_crossfeaturegenerator_fit_categorical_combo_uses_custom_separator():
    """Tests that fit builds combo column names via a custom separator.

    Asserts:
        - Combo name "A-B" is present in encoding_maps_ for separator="-"
        - Combined values inside the encoding map use the custom separator too
    """
    df = pd.DataFrame({"A": ["a", "b"], "B": ["x", "y"]})
    generator = CrossFeatureGenerator(
        features_names=["A", "B"], degree=2, separator="-"
    )
    generator.fit(df)
    if "A-B" not in generator.encoding_maps_:
        raise AssertionError()
    if set(generator.encoding_maps_["A-B"]) != {"a-x", "b-y"}:
        raise AssertionError()


def test_crossfeaturegenerator_fit_numerical_combo_tracked_separately(mixed_dataframe):
    """Tests fit tracks all-numeric feature combinations in numerical_combos_.

    Args:
        mixed_dataframe (pd.DataFrame): Fixture DataFrame with mixed dtypes.

    Asserts:
        - Numeric combo "C_D" is present in numerical_combos_
        - Numeric combo does not receive an encoding map
    """
    generator = CrossFeatureGenerator(features_names=["C", "D"], degree=2)
    generator.fit(mixed_dataframe)
    if "C_D" not in generator.numerical_combos_:
        raise AssertionError()
    if "C_D" in generator.encoding_maps_:
        raise AssertionError()


def test_crossfeaturegenerator_transform_categorical_combo_columns(mixed_dataframe):
    """Tests transform adds an integer-coded column for a categorical combination.

    Args:
        mixed_dataframe (pd.DataFrame): Fixture DataFrame with mixed dtypes.

    Asserts:
        - New combo column is added alongside the original columns
        - Combo column values are integer-coded and correctly mapped
        - Row count is preserved
    """
    generator = CrossFeatureGenerator(features_names=["A", "B"], degree=2)
    generator.fit(mixed_dataframe)
    result = generator.transform(mixed_dataframe)
    if "A_B" not in result.columns:
        raise AssertionError()
    if "A" not in result.columns or "B" not in result.columns:
        raise AssertionError()
    if len(result) != len(mixed_dataframe):
        raise AssertionError()
    expected_codes = [
        generator.encoding_maps_["A_B"][f"{a}_{b}"]
        for a, b in zip(mixed_dataframe["A"], mixed_dataframe["B"])
    ]
    if result["A_B"].tolist() != expected_codes:
        raise AssertionError()


def test_crossfeaturegenerator_transform_numerical_combo_is_elementwise_product(
    mixed_dataframe,
):
    """Tests transform multiplies numeric feature combinations element-wise.

    Args:
        mixed_dataframe (pd.DataFrame): Fixture DataFrame with mixed dtypes.

    Asserts:
        - Combo column values equal the element-wise product of the source columns
    """
    generator = CrossFeatureGenerator(features_names=["C", "D"], degree=2)
    generator.fit(mixed_dataframe)
    result = generator.transform(mixed_dataframe)
    expected = (
        (mixed_dataframe["C"] * mixed_dataframe["D"]).astype("float32").to_numpy()
    )
    if not np.allclose(result["C_D"].to_numpy(), expected):
        raise AssertionError()


def test_crossfeaturegenerator_transform_unseen_categorical_combo_maps_to_negative_one():
    """Tests transform encodes an unseen categorical combination as -1.

    Asserts:
        - A combination never seen during fit is encoded as -1 at transform time
        - A combination seen during fit keeps its fitted encoding
    """
    train_df = pd.DataFrame({"A": ["a", "b"], "B": ["x", "y"]})
    test_df = pd.DataFrame({"A": ["c", "a"], "B": ["z", "x"]})
    generator = CrossFeatureGenerator(features_names=["A", "B"], degree=2)
    generator.fit(train_df)
    result = generator.transform(test_df)
    if result["A_B"].tolist()[0] != -1:
        raise AssertionError()
    if result["A_B"].tolist()[1] != generator.encoding_maps_["A_B"]["a_x"]:
        raise AssertionError()


def test_crossfeaturegenerator_transform_before_fit_raises_value_error():
    """Tests transform raises ValueError when the generator has not been fitted.

    Asserts:
        - ValueError is raised mentioning that the generator must be fitted first
    """
    df = pd.DataFrame({"A": ["a", "b"], "B": ["x", "y"]})
    generator = CrossFeatureGenerator()
    with pytest.raises(
        ValueError, match="CrossFeatureGenerator must be fitted before transform"
    ):
        generator.transform(df)


def test_crossfeaturegenerator_fit_transform_before_fit_not_needed(mixed_dataframe):
    """Tests fit_transform runs fit then transform and matches calling them separately.

    Args:
        mixed_dataframe (pd.DataFrame): Fixture DataFrame with mixed dtypes.

    Asserts:
        - fit_transform output equals output from separate fit() and transform() calls
    """
    generator_combined = CrossFeatureGenerator(features_names=["A", "B"], degree=2)
    result_combined = generator_combined.fit_transform(mixed_dataframe)

    generator_separate = CrossFeatureGenerator(features_names=["A", "B"], degree=2)
    generator_separate.fit(mixed_dataframe)
    result_separate = generator_separate.transform(mixed_dataframe)

    pd.testing.assert_frame_equal(result_combined, result_separate)


def test_crossfeaturegenerator_get_feature_names_empty_before_fit():
    """Tests get_feature_names returns an empty list before fitting.

    Asserts:
        - Empty list is returned when no combinations have been generated
    """
    generator = CrossFeatureGenerator()
    if generator.get_feature_names() != []:
        raise AssertionError()


@pytest.mark.parametrize(
    ("separator", "expected_names"),
    [
        ("_", ["A_B", "A_C", "B_C"]),
        ("-", ["A-B", "A-C", "B-C"]),
    ],
)
def test_crossfeaturegenerator_get_feature_names_reflects_separator(
    separator: str, expected_names: list[str]
):
    """Tests get_feature_names joins combo tuples using the configured separator.

    Args:
        separator (str): Separator configured on the generator.
        expected_names (list[str]): Expected separator-joined combo names.

    Asserts:
        - Returned feature names match the expected separator-joined combo names
    """
    df = pd.DataFrame({"A": ["a", "b"], "B": ["x", "y"], "C": ["1", "2"]})
    generator = CrossFeatureGenerator(
        features_names=["A", "B", "C"], degree=2, separator=separator
    )
    generator.fit(df)
    if sorted(generator.get_feature_names()) != sorted(expected_names):
        raise AssertionError()


def test_crossfeaturegenerator_degree_three_combinations():
    """Tests CrossFeatureGenerator with degree=3 generates 3-way combinations.

    Asserts:
        - Correct number of 3-way combinations are generated (4 choose 3 = 4)
        - Every combination has exactly 3 features
        - Combo column names join 3 feature names with the separator
    """
    df = pd.DataFrame(
        {"A": ["a", "b"], "B": ["x", "y"], "C": ["1", "2"], "D": ["p", "q"]}
    )
    generator = CrossFeatureGenerator(features_names=["A", "B", "C", "D"], degree=3)
    result = generator.fit_transform(df)
    if len(generator.feature_combinations_) != 4:
        raise AssertionError()
    if not all(len(combo) == 3 for combo in generator.feature_combinations_):
        raise AssertionError()
    if "A_B_C" not in result.columns:
        raise AssertionError()


if __name__ == "__main__":
    print("Run this file with pytest to execute tests.")
