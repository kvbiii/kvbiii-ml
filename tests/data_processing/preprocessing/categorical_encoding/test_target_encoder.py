import numpy as np
import pandas as pd
import pytest

from kvbiii_ml.data_processing.preprocessing.categorical_encoding.target_encoder import (
    TargetEncoderWithOriginal,
)


@pytest.fixture
def target_encoder_data() -> tuple[pd.DataFrame, pd.Series]:
    """Provides a mixed-dtype DataFrame with a binary target for target encoding.

    Returns:
        tuple[pd.DataFrame, pd.Series]: Feature matrix with a categorical "city"
            column and a numeric "revenue" column, plus a binary target.
    """
    rng = np.random.default_rng(42)
    n_rows = 60
    city = rng.choice(["Warsaw", "Krakow", "Gdansk"], size=n_rows)
    X = pd.DataFrame(
        {
            "city": city,
            "revenue": rng.exponential(scale=100.0, size=n_rows),
        }
    )
    city_event_rate = {"Warsaw": 0.8, "Krakow": 0.5, "Gdansk": 0.2}
    y = pd.Series(
        rng.random(n_rows) < np.array([city_event_rate[c] for c in city]),
        name="target",
    ).astype(int)
    return X, y


def test_targetencoderwithoriginal_defaults_to_non_numeric_columns_only(
    target_encoder_data,
):
    """Tests that auto-detection (variables=None) selects only non-numeric columns.

    Args:
        target_encoder_data: Fixture mixing a categorical and a numeric column.

    Asserts:
        - variables_ contains "city" but not the numeric "revenue" column.
    """
    X, y = target_encoder_data
    encoder = TargetEncoderWithOriginal()
    encoder.fit(X, y)
    if encoder.variables_ != ["city"]:
        raise AssertionError(f"Expected ['city'], got {encoder.variables_}.")


def test_targetencoderwithoriginal_fit_transform_uses_cross_fitting(
    target_encoder_data,
):
    """Tests that fit_transform() uses scikit-learn's own cross-fitted encoding.

    A naive fit().transform() on the same rows used for fitting would encode
    each row using statistics that include that row's own target, which is
    exactly the leakage scikit-learn's TargetEncoder.fit_transform() avoids via
    internal cross-fitting. This asserts the wrapper's fit_transform() actually
    goes through that cross-fitted path rather than silently falling back to
    fit().transform() (the default sklearn TransformerMixin behavior), by
    checking its output differs from the wrapper's own separate fit().transform()
    call. It does not compare against an independently-constructed raw
    TargetEncoder, since TargetEncoderWithOriginal exposes no random_state and
    two independently-shuffled cross-fits are not expected to match bit-for-bit.

    Args:
        target_encoder_data: Fixture with a categorical column and binary target.

    Asserts:
        - fit_transform()'s encoded "city" column differs from the same
          wrapper class's separate fit().transform() output, proving
          cross-fitting (not plain fit-then-transform) drove fit_transform().
    """
    X, y = target_encoder_data
    variables = ["city"]

    wrapper_fit_transform = TargetEncoderWithOriginal(
        variables=variables
    ).fit_transform(X, y)["city_PREPROCESS_TARGET_ENC"]
    plain_fit_then_transform = (
        TargetEncoderWithOriginal(variables=variables)
        .fit(X, y)
        .transform(X)["city_PREPROCESS_TARGET_ENC"]
    )
    if np.allclose(
        wrapper_fit_transform.to_numpy(), plain_fit_then_transform.to_numpy()
    ):
        raise AssertionError(
            "fit_transform() should differ from fit().transform() due to cross-fitting."
        )


if __name__ == "__main__":
    print("Run this file with pytest to execute tests.")
