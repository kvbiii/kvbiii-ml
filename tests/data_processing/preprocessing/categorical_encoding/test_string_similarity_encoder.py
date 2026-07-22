import pandas as pd
import pytest

from kvbiii_ml.data_processing.preprocessing.categorical_encoding.string_similarity_encoder import (
    StringSimilarityEncoderWithOriginal,
)


@pytest.fixture
def string_similarity_data() -> pd.DataFrame:
    """Provides a categorical column with several distinct training strings.

    Returns:
        pd.DataFrame: DataFrame with a single `product` categorical column
            containing four distinct string values.
    """
    return pd.DataFrame(
        {
            "product": [
                "apple",
                "orange",
                "apple_juice",
                "orange_juice",
                "apple",
                "grape",
            ]
        }
    )


def test_stringsimilarityencoderwithoriginal_fit_returns_self(string_similarity_data):
    """Tests that fit() returns the encoder instance for scikit-learn compatibility.

    Args:
        string_similarity_data: DataFrame with several distinct training strings.

    Asserts:
        - fit() returns the same encoder instance it was called on.
    """
    encoder = StringSimilarityEncoderWithOriginal(variables=["product"])
    result = encoder.fit(string_similarity_data)
    if result is not encoder:
        raise AssertionError("fit() did not return self.")


def test_stringsimilarityencoderwithoriginal_transform_expands_one_column_per_training_string(
    string_similarity_data,
):
    """Tests the 1-to-many expansion contract that diverges from the base class.

    Unlike the other `WithOriginal` wrappers, this class appends one column per
    unique training string per variable, not a single `{variable}_{suffix}` column.

    Args:
        string_similarity_data: DataFrame with several distinct training strings.

    Asserts:
        - The number of newly appended columns equals the number of unique
          training strings, and is greater than one.
        - Original columns are preserved unchanged.
    """
    encoder = StringSimilarityEncoderWithOriginal(variables=["product"])
    encoder.fit(string_similarity_data)
    result = encoder.transform(string_similarity_data)
    unique_training_strings = string_similarity_data["product"].nunique()
    new_columns = [c for c in result.columns if c not in string_similarity_data.columns]
    if len(new_columns) != unique_training_strings:
        raise AssertionError(
            f"Expected {unique_training_strings} new columns, got {len(new_columns)}."
        )
    if len(new_columns) <= 1:
        raise AssertionError(
            "Expected a 1-to-many expansion with more than one new column."
        )
    pd.testing.assert_frame_equal(
        result[["product"]], string_similarity_data[["product"]]
    )


def test_stringsimilarityencoderwithoriginal_transform_before_fit_raises_attributeerror(
    string_similarity_data,
):
    """Tests that transform() before fit() raises AttributeError.

    Args:
        string_similarity_data: DataFrame with several distinct training strings.

    Asserts:
        - Calling transform() on an unfitted encoder raises AttributeError.
    """
    encoder = StringSimilarityEncoderWithOriginal(variables=["product"])
    with pytest.raises(AttributeError):
        encoder.transform(string_similarity_data)


def test_stringsimilarityencoderwithoriginal_get_feature_names_out_reflects_expanded_columns(
    string_similarity_data,
):
    """Tests that get_feature_names_out() reflects the expanded per-string columns.

    Args:
        string_similarity_data: DataFrame with several distinct training strings.

    Asserts:
        - The number of feature names starting with `product_` matches the
          number of unique training strings.
    """
    encoder = StringSimilarityEncoderWithOriginal(variables=["product"])
    encoder.fit(string_similarity_data)
    encoder.transform(string_similarity_data)
    feature_names = encoder.get_feature_names_out()
    expected_new_columns = string_similarity_data["product"].nunique()
    matching = [name for name in feature_names if name.startswith("product_")]
    if len(matching) != expected_new_columns:
        raise AssertionError(
            f"Expected {expected_new_columns} feature names starting with 'product_', "
            f"got {len(matching)}."
        )


if __name__ == "__main__":
    print("Run this file with pytest to execute tests.")
