import warnings
from typing import Any, Literal

import pandas as pd
from sklearn.preprocessing import TargetEncoder

from kvbiii_ml.data_processing.preprocessing.expansion_base import (
    _WithOriginalSubsetBase,
    _cast_numeric_variables_to_object,
)


class TargetEncoderWithOriginal(_WithOriginalSubsetBase):
    """Wraps scikit-learn TargetEncoder to keep originals and append target-encoded copies.

    Derived columns are named ``{original}_PREPROCESS_TARGET_ENC``. Each category is
    replaced by a shrinkage-smoothed estimate of the target - unlike this package's
    feature_engine-based MeanEncoder, scikit-learn's TargetEncoder both smooths
    per-category means toward the global mean (``smooth``) and, on
    ``fit_transform``, internally cross-fits across ``cv`` folds so the
    training-set encoding for each row is not computed from that row's own
    target (avoiding the leakage a naive fit-then-transform on the same data
    would cause). A later, separate ``fit()`` followed by ``transform()`` on
    unseen data instead uses the plain (non-cross-fitted) statistics from the
    full training fit, matching scikit-learn's own documented TargetEncoder
    behavior. Original categorical columns are preserved.

    scikit-learn's TargetEncoder already accepts numeric columns natively, but
    any resolved ``variables_`` that are numeric are still recast to object
    dtype internally before encoding, keeping this wrapper's handling of an
    explicitly-named numeric, low-cardinality column consistent with the other
    categorical encoders in this package.
    """

    _suffix = "PREPROCESS_TARGET_ENC"

    def __init__(
        self,
        categories: Literal["auto"] | list[list] = "auto",
        target_type: Literal["auto", "continuous", "binary", "multiclass"] = "auto",
        smooth: float | Literal["auto"] = "auto",
        cv: int = 5,
        variables: list[str] | None = None,
        random_state: int | None = None,
    ) -> None:
        """
        Initialize TargetEncoderWithOriginal.

        Args:
            categories (Literal["auto"] | list[list], optional): Per-variable
                category lists, or ``"auto"`` to detect them during fitting.
                Defaults to "auto".
            target_type (Literal["auto", "continuous", "binary", "multiclass"], optional):
                Kind of target being encoded against, or ``"auto"`` to infer it
                from y. Defaults to "auto".
            smooth (float | Literal["auto"], optional): Shrinkage strength toward
                the global target mean - higher smooths harder; ``"auto"`` picks
                it from the target's variance. Defaults to "auto".
            cv (int, optional): Number of cross-fitting folds used to encode the
                training set inside ``fit_transform``. Defaults to 5.
            variables (list[str] | None, optional): Columns to encode. May include
                numeric columns - they are recast to object dtype internally
                before encoding. Defaults to None (auto-detect all non-numeric
                columns).
            random_state (int | None, optional): Seed controlling the shuffle
                scikit-learn's ``TargetEncoder`` applies before splitting into
                ``cv`` folds for its internal cross-fitting. Left unset, that
                shuffle uses a fresh random split on every single
                ``fit_transform`` call, so encoded values - and therefore any
                metric computed downstream of them - differ between otherwise
                identical runs even with every other seed fixed. Passing a
                fixed value here removes that source of non-determinism.
                Defaults to None, which reproduces the unseeded behavior
                exactly.
        """
        self.categories = categories
        self.target_type = target_type
        self.smooth = smooth
        self.cv = cv
        self.variables = variables
        self.random_state = random_state

    def _build_inner(self) -> TargetEncoder:
        """Return a fresh TargetEncoder configured from instance attributes.

        ``random_state`` is only forwarded when explicitly set, since scikit-learn's
        ``TargetEncoder`` emits a ``FutureWarning`` on every call that passes it (its
        deprecated replacement for the ``shuffle``/``random_state`` pair) - callers
        that don't need reproducibility keep the original, warning-free behavior.

        Returns:
            TargetEncoder: Unfitted encoder instance.
        """
        inner_kwargs: dict[str, Any] = {
            "categories": self.categories,
            "target_type": self.target_type,
            "smooth": self.smooth,
            "cv": self.cv,
        }
        if self.random_state is not None:
            inner_kwargs["random_state"] = self.random_state
        return TargetEncoder(**inner_kwargs)

    def _default_variables(self, X: pd.DataFrame) -> list[str]:
        """Auto-detect non-numeric columns to encode when ``variables`` is None.

        Args:
            X (pd.DataFrame): Training features.

        Returns:
            list[str]: Names of non-numeric columns in X.
        """
        return [c for c in X.columns if not pd.api.types.is_numeric_dtype(X[c])]

    def _prepare_variables(self, X: pd.DataFrame) -> pd.DataFrame:
        """Recast any numeric `variables_` to object dtype before encoding.

        Args:
            X (pd.DataFrame): Full input features, not yet subset.

        Returns:
            pd.DataFrame: X with any numeric `variables_` recast to object dtype.
        """
        return _cast_numeric_variables_to_object(X, self.variables_)

    def fit_transform(self, X: pd.DataFrame, y: Any = None) -> pd.DataFrame:
        """Fit using cross-fitted target statistics and transform in one pass.

        Overrides the inherited fit().transform() so the training set is
        encoded via scikit-learn's internal cross-fitting (see class
        docstring), rather than with the same estimator later reused for
        out-of-sample transform() calls.

        Args:
            X (pd.DataFrame): Training features.
            y (Any): Target used to fit the per-category encodings.

        Returns:
            pd.DataFrame: Original columns plus one cross-fitted encoded column
                per encoded variable.
        """
        self.variables_ = self._resolve_variables(X)
        self._inner = self._build_inner()
        prepared = self._prepare_variables(X)[self.variables_]
        if self.random_state is not None:
            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore",
                    message=r".*TargetEncoder\.shuffle.*",
                    category=FutureWarning,
                )
                encoded = self._inner.fit_transform(prepared, y)
        else:
            encoded = self._inner.fit_transform(prepared, y)
        return pd.concat([X.copy(), self._derived_frame(encoded, X.index)], axis=1)


__all__ = ["TargetEncoder", "TargetEncoderWithOriginal"]


if __name__ == "__main__":
    import numpy as np

    def _run_demo() -> None:
        """Run TargetEncoder and its original-preserving variant."""
        rng = np.random.default_rng(42)
        df = pd.DataFrame(
            {
                "city": rng.choice(["Warsaw", "Krakow", "Gdansk"], size=300),
                "tier": rng.choice(["A", "B", "C"], size=300),
                "revenue": rng.exponential(150.0, size=300),
            }
        )
        y = pd.Series((df["revenue"] > 150.0).astype(int), name="target")

        enc = TargetEncoder(target_type="binary")
        print("=== TargetEncoder (replace) ===")
        print(enc.fit_transform(df[["city", "tier"]], y)[:5])

        enc_exp = TargetEncoderWithOriginal(
            variables=["city", "tier"], target_type="binary"
        )
        print("\n=== TargetEncoderWithOriginal (append, cross-fitted) ===")
        print(enc_exp.fit_transform(df[["city", "tier", "revenue"]], y).head())

    _run_demo()
