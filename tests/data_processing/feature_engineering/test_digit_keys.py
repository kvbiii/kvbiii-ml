"""Tests for digit-key feature generation and its target-encoding recipe."""

import numpy as np
import pandas as pd
from feature_engine.selection import DropFeatures
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline

from kvbiii_ml.data_processing.feature_engineering.digit_keys import DigitKeysFeatureGenerator
from kvbiii_ml.data_processing.preprocessing.categorical_encoding.target_encoder import (
    TargetEncoderWithOriginal,
)
from kvbiii_ml.modeling.training.cross_validation import CrossValidationTrainer


def test_digit_keys_on_known_floats() -> None:
    """Keys of known floats, including ones whose x*100 or x*10 is not exact in binary."""
    frame = pd.DataFrame({"x": [61.16, 9.77, 0.5, 1234.05, 0.29, np.nan]})

    keys = DigitKeysFeatureGenerator(["x"]).fit_transform(frame)

    expected = {
        "CENTS": [16, 77, 50, 5, 29, -1],
        "INT": [61, 9, 0, 1234, 0, -1],
        "TEN": [6, 0, 0, 123, 0, -1],
        "DEC": [611, 97, 5, 12340, 2, -1],
        "LENGTH": [2, 1, 0, 4, 0, -1],
        "LEAD1": [26, 19, 5, 41, 2, -1],
        "LEAD2": [261, 197, 50, 412, 29, -1],
    }
    for key, values in expected.items():
        column = keys[f"x_PREPROCESS_KEY_{key}"]
        if column.tolist() != values or column.dtype != np.int64:
            raise AssertionError((key, column.tolist()))
    if keys["x"].isna().sum() != 1:
        raise AssertionError("original column must be kept unchanged")


def test_digit_keys_subset_and_feature_names() -> None:
    """Only the requested keys are added, named in key order after the originals."""
    frame = pd.DataFrame({"x": [1.25], "y": [3.5]})
    generator = DigitKeysFeatureGenerator(["x", "y"], keys=("cents", "lead1")).fit(frame)

    names = list(generator.get_feature_names_out())
    if names != [
        "x",
        "y",
        "x_PREPROCESS_KEY_CENTS",
        "x_PREPROCESS_KEY_LEAD1",
        "y_PREPROCESS_KEY_CENTS",
        "y_PREPROCESS_KEY_LEAD1",
    ]:
        raise AssertionError(names)
    if list(generator.transform(frame).columns) != names:
        raise AssertionError()


def test_digit_key_target_encoding_canary_gives_chance_auc_on_shuffled_labels() -> None:
    """Keys + cross-fitted m-estimate TE inside CV must not leak: shuffled labels give AUC ~0.5."""
    rng = np.random.default_rng(0)
    n = 40_000
    frame = pd.DataFrame({"amount": np.round(rng.lognormal(3, 1, n), 2)})
    y = pd.Series(rng.permutation((rng.uniform(size=n) < 0.3).astype(int)))
    keys = [f"amount_PREPROCESS_KEY_{k}" for k in ("INT", "TEN", "DEC", "LEAD1", "LEAD2", "LENGTH")]
    encoded = [f"{c}_PREPROCESS_TARGET_ENC" for c in ["amount", *keys]]
    pipeline = Pipeline(
        [
            ("keys", DigitKeysFeatureGenerator(["amount"])),
            (
                "te",
                TargetEncoderWithOriginal(smooth=20, variables=["amount", *keys], random_state=0),
            ),
            ("drop", DropFeatures(["amount", *keys])),
        ]
    )
    trainer = CrossValidationTrainer(
        "classification",
        "Roc AUC",
        cv=StratifiedKFold(5, shuffle=True, random_state=0),
        preprocessing_pipeline=pipeline,
        verbose=False,
    )

    _, valid_scores, _ = trainer.fit(LogisticRegression(), frame, y)

    fitted = trainer.fitted_pipelines_[0]
    if fitted is None or not set(encoded) <= set(fitted.transform(frame.head()).columns):
        raise AssertionError("TE columns missing")
    if abs(float(np.mean(valid_scores)) - 0.5) > 0.01:
        raise AssertionError(valid_scores)
