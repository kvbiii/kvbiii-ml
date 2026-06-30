import abc
import contextlib
import warnings
from typing import Any

import pandas as pd
from feature_engine.outliers import Winsorizer
from sklearn.base import BaseEstimator, TransformerMixin


class _WithOriginalBase(BaseEstimator, TransformerMixin, abc.ABC):
    """Abstract base for transformers that keep originals and append derived columns.

    Subclasses declare two class-level attributes and implement three hook methods.
    The base handles column concatenation so derived columns are named
    ``{source_col}_{_suffix}``. Original columns are always preserved, making
    this safe to compose with sklearn Pipelines that track feature names.

    Class-level attributes to set in every subclass:
        _suffix (str): Non-empty string appended to each derived column name.
        _suppress_warnings (bool): When True, suppresses the noisy feature_engine
            ``UserWarning`` about datetime format inference during fit and
            transform. Set to True for any feature_engine encoder that triggers
            it. Defaults to False.

    Abstract methods to implement in every subclass:
        _build_inner: Return a fresh unfitted inner transformer.
        _fit_inner: Fit the inner transformer and return the fitted instance.
        _transform_inner: Apply the fitted inner transformer and return the result.
    """

    _suffix: str = ""
    _suppress_warnings: bool = False

    @abc.abstractmethod
    def _build_inner(self) -> Any:
        """Return a fresh, unfitted inner transformer.

        Returns:
            Any: Unfitted inner transformer instance.
        """

    @abc.abstractmethod
    def _fit_inner(self, inner: Any, X: pd.DataFrame, y: Any) -> Any:
        """Fit the inner transformer on X (and optionally y).

        Args:
            inner (Any): Unfitted inner transformer from _build_inner.
            X (pd.DataFrame): Training features.
            y (Any): Target forwarded from fit(). Pass to inner.fit() for
                supervised transformers; ignore it for unsupervised ones.

        Returns:
            Any: Fitted inner transformer.
        """

    @abc.abstractmethod
    def _transform_inner(self, inner: Any, X: pd.DataFrame) -> pd.DataFrame:
        """Apply the fitted inner transformer to X.

        Args:
            inner (Any): Fitted inner transformer.
            X (pd.DataFrame): Features to transform.

        Returns:
            pd.DataFrame: Transformed features.
        """

    def fit(self, X: pd.DataFrame, y: Any = None) -> "_WithOriginalBase":
        """Fit the inner transformer on X.

        Args:
            X (pd.DataFrame): Training features.
            y (Any, optional): Target forwarded to supervised inner transformers.
                Defaults to None.

        Returns:
            _WithOriginalBase: Fitted instance (self).
        """
        inner = self._build_inner()
        with self._maybe_suppress_warnings():
            self._inner = self._fit_inner(inner, X.copy(), y)
        self.variables_: list[str] = getattr(self._inner, "variables_", [])
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Append derived columns to X, preserving all original columns.

        Derived column names follow the pattern ``{source_col}_{_suffix}``.

        Args:
            X (pd.DataFrame): Features to transform.

        Returns:
            pd.DataFrame: Original columns plus one derived column per encoded variable.
        """
        X_orig = X.copy()
        with self._maybe_suppress_warnings():
            X_transformed = self._transform_inner(self._inner, X.copy())
        new_cols = {
            f"{var}_{self._suffix}": X_transformed[var] for var in self.variables_
        }
        return pd.concat([X_orig, pd.DataFrame(new_cols, index=X_orig.index)], axis=1)

    @contextlib.contextmanager
    def _maybe_suppress_warnings(self):
        """Context manager that activates warning suppression when _suppress_warnings is True."""
        if self._suppress_warnings:
            with self._suppress_fe_datetime_warnings():
                yield
        else:
            yield

    @staticmethod
    @contextlib.contextmanager
    def _suppress_fe_datetime_warnings():
        """Suppress the feature_engine UserWarning about datetime format inference.

        The warning fires in feature_engine/variable_handling/_variable_type_checks.py
        when feature_engine attempts pd.to_datetime without a format string. It is
        harmless - feature_engine falls back to dateutil parsing automatically.
        """
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message="Could not infer format",
                category=UserWarning,
                module=r"feature_engine\.variable_handling\._variable_type_checks",
            )
            yield
