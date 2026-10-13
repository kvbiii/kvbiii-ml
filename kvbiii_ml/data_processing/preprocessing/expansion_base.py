import abc
import contextlib
import warnings
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


def _cast_numeric_variables_to_object(
    X: pd.DataFrame, variables: list[str] | None
) -> pd.DataFrame:
    """Recast any numeric columns among `variables` to object dtype.

    Several encoders in this package only accept object/categorical input and
    reject numeric columns outright (e.g. feature_engine's CountFrequencyEncoder
    raises "Some of the variables are not categorical..."). This lets a caller
    name a numeric, low-cardinality column (an already-coded category, a small
    integer flag, ...) directly in `variables` without pre-casting their raw
    data. Only the explicitly requested `variables` are touched, and only when
    numeric, so a `variables=None` auto-detect path - which typically only ever
    picks up already non-numeric columns - is unaffected by this helper.

    Args:
        X (pd.DataFrame): Features to adjust. Not mutated - a copy is made
            only when a cast is actually needed.
        variables (list[str] | None): Columns to force to object dtype where
            numeric, or None/empty to leave X untouched.

    Returns:
        pd.DataFrame: X (or a copy, if any column was recast) with every
            numeric column in `variables` now object dtype.
    """
    if not variables:
        return X
    numeric_variables = [v for v in variables if pd.api.types.is_numeric_dtype(X[v])]
    if not numeric_variables:
        return X
    X = X.copy()
    for var in numeric_variables:
        X[var] = X[var].astype(object)
    return X


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
        x_orig = X.copy()
        with self._maybe_suppress_warnings():
            x_transformed = self._transform_inner(self._inner, X.copy())
        new_cols = {
            f"{var}_{self._suffix}": x_transformed[var] for var in self.variables_
        }
        return pd.concat([x_orig, pd.DataFrame(new_cols, index=x_orig.index)], axis=1)

    def get_derived_column_dependencies(self) -> dict[str, list[str]]:
        """Map each derived output column this transformer adds to its source column(s).

        Only covers the new columns transform() appends - pass-through columns are
        handled generically by any consumer walking a pipeline, for every step
        regardless of whether it implements this method. Mirrors transform()'s own
        naming exactly (reads self.variables_ and self._suffix), so it can never
        drift from what transform() actually produces. Must be called after fit().

        Returns:
            dict[str, list[str]]: Derived column name mapped to a single-element
                list containing the raw/source column name it depends on.
        """
        return {f"{var}_{self._suffix}": [var] for var in self.variables_}

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


class _WithOriginalSubsetBase(BaseEstimator, TransformerMixin, abc.ABC):
    """Abstract base for transformers that keep originals and append derived columns
    computed from an inner transformer restricted to an explicit column subset.

    Built for raw scikit-learn transformers (e.g. ``KBinsDiscretizer``,
    ``TargetEncoder``) that, unlike the feature_engine transformers `_WithOriginalBase`
    is built for, only accept and return exactly the columns they are given, as a
    bare numpy array with no column labels - they cannot be fit on the full input
    frame and do not preserve passthrough columns. This base instead resolves
    which `variables` to transform (explicit, or auto-detected via
    `_default_variables`), fits/transforms only that subset, and reattaches the
    result as named ``{source_col}_{_suffix}`` columns onto the untouched
    original DataFrame.

    Class-level attributes to set in every subclass:
        _suffix (str): Non-empty string appended to each derived column name.
            May be defined as a computed ``@property`` when the suffix should
            depend on constructor parameters (e.g. a bin count).
        _suppress_warnings (bool): When True, suppresses scikit-learn's noisy
            ``UserWarning`` about ``KBinsDiscretizer`` collapsing bin edges
            that are too close together. Set to True for any inner
            transformer that triggers it. Defaults to False.

    Abstract methods to implement in every subclass:
        _build_inner: Return a fresh unfitted inner transformer.
        _default_variables: Auto-detect columns to transform when ``variables`` is None.
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
    def _default_variables(self, X: pd.DataFrame) -> list[str]:
        """Auto-detect which columns to transform when ``variables`` is None.

        Args:
            X (pd.DataFrame): Training features.

        Returns:
            list[str]: Column names to fit/transform.
        """

    def fit(self, X: pd.DataFrame, y: Any = None) -> "_WithOriginalSubsetBase":
        """Resolve the target variables and fit the inner transformer on that subset.

        Args:
            X (pd.DataFrame): Training features.
            y (Any, optional): Target forwarded to supervised inner transformers.
                Defaults to None.

        Returns:
            _WithOriginalSubsetBase: Fitted instance (self).
        """
        self.variables_ = self._resolve_variables(X)
        with self._maybe_suppress_warnings():
            self._inner = self._build_inner().fit(
                self._prepare_variables(X)[self.variables_], y
            )
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Append derived columns to X, preserving all original columns.

        Args:
            X (pd.DataFrame): Features to transform.

        Returns:
            pd.DataFrame: Original columns plus one derived column per variable.
        """
        encoded = self._inner.transform(self._prepare_variables(X)[self.variables_])
        return pd.concat([X.copy(), self._derived_frame(encoded, X.index)], axis=1)

    def get_derived_column_dependencies(self) -> dict[str, list[str]]:
        """Map each derived output column this transformer adds to its source column.

        Mirrors transform()'s own naming exactly, so it can never drift from
        what transform() actually produces. Must be called after fit().

        Returns:
            dict[str, list[str]]: Derived column name mapped to a single-element
                list containing the raw/source column name it depends on.
        """
        return {f"{var}_{self._suffix}": [var] for var in self.variables_}

    def _derived_frame(self, encoded: np.ndarray, index: pd.Index) -> pd.DataFrame:
        """Wrap a bare ``(n_samples, n_variables)`` array as named derived columns.

        Args:
            encoded (np.ndarray): Inner transformer output for `self.variables_`.
            index (pd.Index): Row index to align the derived columns to.

        Returns:
            pd.DataFrame: One named ``{var}_{_suffix}`` column per variable, in
                `self.variables_` order.
        """
        columns = [f"{var}_{self._suffix}" for var in self.variables_]
        return pd.DataFrame(encoded, columns=columns, index=index)

    def _prepare_variables(self, X: pd.DataFrame) -> pd.DataFrame:
        """Adjust X immediately before it is subset to `variables_`. No-op by default.

        Hook for subclasses whose inner transformer needs `variables_` adjusted
        (e.g. dtype-coerced) beyond plain column selection, without having to
        duplicate fit()/transform() themselves.

        Args:
            X (pd.DataFrame): Full input features, not yet subset.

        Returns:
            pd.DataFrame: Adjusted features, same shape as X.
        """
        return X

    def _resolve_variables(self, X: pd.DataFrame) -> list[str]:
        """Resolve the columns to transform: explicit `variables`, else auto-detect.

        Args:
            X (pd.DataFrame): Training features.

        Returns:
            list[str]: Column names to fit/transform.
        """
        if self.variables is not None:
            return list(self.variables)
        return self._default_variables(X)

    @contextlib.contextmanager
    def _maybe_suppress_warnings(self):
        """Context manager that activates warning suppression when _suppress_warnings is True."""
        if self._suppress_warnings:
            with self._suppress_kbins_benign_warnings():
                yield
        else:
            yield

    @staticmethod
    @contextlib.contextmanager
    def _suppress_kbins_benign_warnings():
        """Suppress KBinsDiscretizer's UserWarnings for its own benign fallbacks.

        Both fire in sklearn/preprocessing/_discretization.py and are harmless -
        KBinsDiscretizer already falls back on its own in either case:

        - "Bins whose width are too small" - tied or near-duplicate values leave
          fewer distinct bin edges than the requested ``n_bins`` (e.g. a
          low-cardinality or heavily-skewed column).
        - "is constant and will be replaced with 0" - a column has a single
          distinct value, so it is collapsed into one bin.
        """
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message="Bins whose width are too small",
                category=UserWarning,
                module=r"sklearn\.preprocessing\._discretization",
            )
            warnings.filterwarnings(
                "ignore",
                message=r"Feature \d+ is constant and will be replaced with 0\.",
                category=UserWarning,
                module=r"sklearn\.preprocessing\._discretization",
            )
            yield
