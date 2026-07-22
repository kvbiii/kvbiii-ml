import pandas as pd
import pytest

from kvbiii_ml.data_processing.preprocessing.categorical_encoding.rare_label_encoder import (
    RareLabelEncoderWithOriginal,
)


@pytest.fixture
def rare_label_data() -> pd.DataFrame:
    """Provides a categorical column with clearly rare and clearly frequent categories.

    Out of 10 rows: Warsaw has frequency 0.6, Krakow 0.2, Gdansk 0.1, Poznan 0.1.

    Returns:
        pd.DataFrame: DataFrame with a single `city` categorical column.
    """
    return pd.DataFrame(
        {"city": ["Warsaw"] * 6 + ["Krakow"] * 2 + ["Gdansk"] * 1 + ["Poznan"] * 1}
    )


def test_rarelabelencoderwithoriginal_transform_groups_categories_below_tol(
    rare_label_data,
):
    """Tests that categories with frequency below tol are grouped into the rare label.

    Args:
        rare_label_data: DataFrame with clearly rare and clearly frequent categories.

    Asserts:
        - Categories below the tol threshold are replaced with the default "Rare" label.
        - Categories above the tol threshold retain their original label.
    """
    encoder = RareLabelEncoderWithOriginal(tol=0.15, n_categories=2, variables=["city"])
    encoder.fit(rare_label_data)
    result = encoder.transform(rare_label_data)
    encoded = result["city_PREPROCESS_RARE"]
    if not (encoded[rare_label_data["city"] == "Gdansk"] == "Rare").all():
        raise AssertionError("Gdansk should have been grouped into 'Rare'.")
    if not (encoded[rare_label_data["city"] == "Poznan"] == "Rare").all():
        raise AssertionError("Poznan should have been grouped into 'Rare'.")
    if not (encoded[rare_label_data["city"] == "Warsaw"] == "Warsaw").all():
        raise AssertionError("Warsaw should have retained its original label.")
    if not (encoded[rare_label_data["city"] == "Krakow"] == "Krakow").all():
        raise AssertionError("Krakow should have retained its original label.")


def test_rarelabelencoderwithoriginal_transform_respects_n_categories_threshold(
    rare_label_data,
):
    """Tests that grouping is skipped when distinct categories are below n_categories.

    Args:
        rare_label_data: DataFrame with clearly rare and clearly frequent categories.

    Asserts:
        - No category is grouped into "Rare" when n_categories exceeds the
          number of distinct categories present.
    """
    encoder = RareLabelEncoderWithOriginal(
        tol=0.15, n_categories=10, variables=["city"]
    )
    encoder.fit(rare_label_data)
    result = encoder.transform(rare_label_data)
    if (result["city_PREPROCESS_RARE"] == "Rare").any():
        raise AssertionError(
            "Grouping occurred despite n_categories threshold not being met."
        )


def test_rarelabelencoderwithoriginal_transform_respects_max_n_categories(
    rare_label_data,
):
    """Tests that max_n_categories caps the number of categories kept as-is.

    Args:
        rare_label_data: DataFrame with clearly rare and clearly frequent categories.

    Asserts:
        - Only the single most frequent category is kept; all others are grouped
          into "Rare" when max_n_categories=1.
    """
    encoder = RareLabelEncoderWithOriginal(
        tol=0.0, n_categories=1, max_n_categories=1, variables=["city"]
    )
    encoder.fit(rare_label_data)
    result = encoder.transform(rare_label_data)
    encoded = result["city_PREPROCESS_RARE"]
    if not (encoded[rare_label_data["city"] == "Warsaw"] == "Warsaw").all():
        raise AssertionError("Warsaw should have remained the sole frequent category.")
    if not (encoded[rare_label_data["city"] != "Warsaw"] == "Rare").all():
        raise AssertionError(
            "Non-Warsaw categories should have been grouped into 'Rare'."
        )


def test_rarelabelencoderwithoriginal_transform_uses_custom_replace_with(
    rare_label_data,
):
    """Tests that replace_with overrides the default "Rare" grouping label.

    Args:
        rare_label_data: DataFrame with clearly rare and clearly frequent categories.

    Asserts:
        - Grouped categories are replaced with the configured custom label.
        - The default "Rare" label does not appear in the output.
    """
    encoder = RareLabelEncoderWithOriginal(
        tol=0.15, n_categories=2, replace_with="Other", variables=["city"]
    )
    encoder.fit(rare_label_data)
    result = encoder.transform(rare_label_data)
    encoded = result["city_PREPROCESS_RARE"]
    if "Rare" in encoded.to_numpy():
        raise AssertionError("Default 'Rare' label leaked despite custom replace_with.")
    if "Other" not in encoded.to_numpy():
        raise AssertionError("Custom replace_with label was not applied.")


if __name__ == "__main__":
    print("Run this file with pytest to execute tests.")
