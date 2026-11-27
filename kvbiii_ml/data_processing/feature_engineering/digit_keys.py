import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

ALL_KEYS = ("cents", "int", "ten", "dec", "lead1", "lead2", "length")
# Rounding before floor() undoes binary float error, e.g. 0.29 * 100 = 28.999999999999996.
_ROUND_DECIMALS = 6


class DigitKeysFeatureGenerator(BaseEstimator, TransformerMixin):
    """Append integer "digit keys" of numeric columns, meant to be target encoded as categories.

    For a value x (with a = |x| for the leading-digit keys):

    - ``cents = round(100x) % 100``
    - ``int = floor(x)``, ``ten = floor(x / 10)``, ``dec = floor(10x)``
    - ``length`` = number of integer digits of a (0 when a < 1)
    - ``lead1 = length * 10 + d1`` and ``lead2 = length * 100 + d1d2``, where d1 and d1d2 are the
      first one and two digits of a counted from its leading integer digit (the first decimals
      when a < 1): 61.16 -> lead1 26, lead2 261; 9.77 -> 19, 197; 0.5 -> 5, 50.

    Derived columns are named ``{original}_PREPROCESS_KEY_{KEY}`` (int64; NaN becomes
    ``fill_value``). Originals are kept. The transformer is stateless.

    Recipe (cross-fitted m-estimate target encoding of exact values and keys, as one
    ``preprocessing_pipeline`` for ``CrossValidationTrainer``)::

        keys = [f"{c}_PREPROCESS_KEY_{k.upper()}" for c in cols for k in ALL_KEYS]
        Pipeline([
            ("keys", DigitKeysFeatureGenerator(cols)),
            ("te", TargetEncoderWithOriginal(smooth=20, cv=5, variables=cols + keys,
                                             random_state=SEED)),
            ("drop", DropFeatures([k for k in keys if not k.endswith("_CENTS")])),
        ])

    ``TargetEncoderWithOriginal(smooth=20)`` is the m-estimate ``(sum_y + 20 prior) / (n + 20)``,
    cross-fitted over ``cv`` folds of the training rows in ``fit_transform`` and fitted on all
    training rows for ``transform``.
    """

    _suffix = "PREPROCESS_KEY"

    def __init__(
        self,
        features_names: list[str],
        keys: tuple[str, ...] = ALL_KEYS,
        fill_value: int = -1,
    ) -> None:
        """
        Initialize the DigitKeysFeatureGenerator.

        Args:
            features_names (list[str]): Numeric columns to derive keys from.
            keys (tuple[str, ...]): Subset of ``ALL_KEYS`` to add, in this order. Defaults to all.
            fill_value (int): Key value for NaN inputs. Defaults to -1.

        Raises:
            ValueError: If a key is not in ``ALL_KEYS``.
        """
        unknown = set(keys) - set(ALL_KEYS)
        if unknown:
            raise ValueError(f"Unknown keys {sorted(unknown)}; choose from {ALL_KEYS}.")
        self.features_names = features_names
        self.keys = keys
        self.fill_value = fill_value

    def fit(self, X: pd.DataFrame, y: object = None) -> "DigitKeysFeatureGenerator":
        """Record the input columns (no statistics are learned).

        Args:
            X (pd.DataFrame): Input features.
            y (object): Ignored.

        Returns:
            DigitKeysFeatureGenerator: self.
        """
        self.feature_names_in_ = np.asarray(X.columns, dtype=object)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Append the key columns.

        Args:
            X (pd.DataFrame): Input features containing ``features_names``.

        Returns:
            pd.DataFrame: X plus one int64 column per (feature, key).
        """
        derived = {
            f"{col}_{self._suffix}_{key.upper()}": values
            for col in self.features_names
            for key, values in self._keys_of(X[col].to_numpy(dtype=float)).items()
        }
        return pd.concat([X, pd.DataFrame(derived, index=X.index)], axis=1)

    def get_feature_names_out(self, input_features: list[str] | None = None) -> np.ndarray:
        """Return the input columns followed by the key columns.

        Args:
            input_features (list[str] | None): Input column names; defaults to those seen in fit.

        Returns:
            np.ndarray: Output column names.
        """
        base = list(self.feature_names_in_ if input_features is None else input_features)
        derived = [
            f"{c}_{self._suffix}_{k.upper()}" for c in self.features_names for k in self.keys
        ]
        return np.asarray(base + derived, dtype=object)

    def _keys_of(self, x: np.ndarray) -> dict[str, np.ndarray]:
        """Compute the requested keys of one column."""
        missing = np.isnan(x)
        x = np.where(missing, 0.0, x)
        a = np.abs(x)

        def floor(values: np.ndarray) -> np.ndarray:
            return np.floor(np.round(values, _ROUND_DECIMALS))

        int_part = floor(a)
        length = np.where(int_part >= 1, np.floor(np.log10(np.maximum(int_part, 1))) + 1, 0)
        formulas = {
            "cents": lambda: np.round(100 * x) % 100,
            "int": lambda: floor(x),
            "ten": lambda: floor(x / 10),
            "dec": lambda: floor(10 * x),
            "length": lambda: length,
            "lead1": lambda: length * 10 + floor(a / 10.0 ** (length - 1)),
            "lead2": lambda: length * 100 + floor(a / 10.0 ** (length - 2)),
        }
        return {
            key: np.where(missing, self.fill_value, formulas[key]()).astype(np.int64)
            for key in self.keys
        }


__all__ = ["ALL_KEYS", "DigitKeysFeatureGenerator"]
