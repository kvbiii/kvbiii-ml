import warnings
from typing import Literal

import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted

DEFAULT_FILL_VALUE = "Unknown"
FILL_STRATEGIES = ("mode", "constant")


MAX_UNKNOWN_PREVIEW = 5


class CategoricalAligner(BaseEstimator, TransformerMixin):
    """Align categorical features between train and test sets using stored train categories."""

    def __init__(
        self,
        variables: list[str] | None = None,
        fill_values: dict[str, str] | None = None,
        fill_strategy: Literal["mode", "constant"] = "constant",
        default_value: str = DEFAULT_FILL_VALUE,
        warn_on_unknown: bool = True,
        categorical_features: list[str] | None = None,
    ) -> None:
        """
        Initialize the CategoricalAligner.

        ``variables`` lets ``build_restricted_pipeline`` narrow the aligned columns.

        Args:
            variables (list[str] | None): Columns to align. None auto-detects them during fit.
            fill_values (dict[str, str] | None): Per-feature fills that override fill_strategy.
            fill_strategy (Literal["mode", "constant"]): Mode of training data, or default_value.
            default_value (str): Constant fill and no-mode fallback. Defaults to "Unknown".
            warn_on_unknown (bool): Warn when unknown categories are found. Defaults to True.
            categorical_features (list[str] | None): Deprecated alias of variables.
        """
        self.variables = variables
        self.fill_values = fill_values
        self.fill_strategy = fill_strategy
        self.default_value = default_value
        self.warn_on_unknown = warn_on_unknown
        self.categorical_features = categorical_features

    @classmethod
    def create_fill_values(
        cls,
        df: pd.DataFrame,
        categorical_features: list[str] | None = None,
        custom_fills: dict[str, str] | None = None,
        default_value: str = DEFAULT_FILL_VALUE,
    ) -> dict[str, str]:
        """
        Helper method to create fill_values dictionary for categorical features.

        Args:
            df (pd.DataFrame): DataFrame to analyze.
            categorical_features (list[str] | None): Features to align. None detects category dtype.
            custom_fills (dict[str, str] | None): Fill values per feature. Defaults to None.
            default_value (str): Fill value for features not in custom_fills. Defaults to "Unknown".

        Returns:
            dict[str, str]: Dictionary mapping feature names to fill values.
        """
        custom_fills = custom_fills or {}
        categorical_features = categorical_features or []
        return {
            feature: custom_fills.get(feature, default_value)
            for feature in categorical_features
            if feature in df.columns
        }

    def fit(self, df: pd.DataFrame, _y: pd.Series | None = None) -> "CategoricalAligner":
        """
        Fit the transformer by learning categories and modes from training data.

        Every fit starts from scratch: ``categorical_features_``, ``categories_``
        and ``modes_`` are rebuilt, and constructor parameters are never modified.

        Args:
            df (pd.DataFrame): Training data.
            _y (pd.Series | None, optional): Target (unused). Defaults to None.

        Returns:
            CategoricalAligner: Fitted transformer.

        Raises:
            ValueError: If fill_strategy is not one of "mode" or "constant".
        """
        if self.fill_strategy not in FILL_STRATEGIES:
            raise ValueError(
                f"fill_strategy must be one of {FILL_STRATEGIES}, got '{self.fill_strategy}'."
            )
        df = df.copy()
        features = self.variables
        if features is None and self.categorical_features is not None:
            warnings.warn(
                "CategoricalAligner(categorical_features=...) is deprecated; use "
                "variables=... instead.",
                FutureWarning,
                stacklevel=2,
            )
            features = self.categorical_features
        if features is None:
            features = df.select_dtypes(include=["object", "string", "category"]).columns.tolist()
        self.categorical_features_ = list(features)
        self.categories_: dict[str, list[str]] = {}
        self.modes_: dict[str, str] = {}
        for feature in self.categorical_features_:
            if feature not in df.columns:
                continue
            mode_series = df[feature].mode(dropna=True)
            self.modes_[feature] = (
                str(mode_series[0]) if not mode_series.empty else self.default_value
            )
            fill_value = self._get_fill_value(feature)
            df[feature] = self._fillna_as_str(df[feature], fill_value).astype("category")
            cats = df[feature].cat.categories.tolist()
            if fill_value not in cats:
                cats = sorted([*cats, fill_value])
            self.categories_[feature] = cats
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Align categories with the fitted ones, raising NotFittedError before fit.

        Args:
            df (pd.DataFrame): Data to transform.

        Returns:
            pd.DataFrame: Transformed data with aligned categories.
        """
        check_is_fitted(self, "categories_")
        df = df.copy()
        for feature, categories in self.categories_.items():
            if feature not in df.columns:
                continue
            fill_value = self._get_fill_value(feature)
            df[feature] = self._fillna_as_str(df[feature], fill_value)
            unknown_mask = ~df[feature].isin(categories)
            unknown_count = unknown_mask.sum()
            if self.warn_on_unknown and unknown_count > 0:
                unknown_values = df.loc[unknown_mask, feature].unique()
                n_unique_unknown = len(unknown_values)
                pct_unknown = unknown_count / len(df) * 100
                unknown_preview = list(unknown_values[:MAX_UNKNOWN_PREVIEW])
                suffix = "..." if n_unique_unknown > MAX_UNKNOWN_PREVIEW else ""
                category_word = "y" if n_unique_unknown == 1 else "ies"
                warnings.warn(
                    (
                        f"\n[{self.__class__.__name__}.transform]\n"
                        f"  Feature '{feature}':\n"
                        f"  Found {unknown_count} rows ({pct_unknown:.2f}%) "
                        f"with {n_unique_unknown} unknown categor{category_word} "
                        "not seen during training.\n"
                        f"  Unknown values: {unknown_preview}{suffix}\n"
                        f"  Replacing with '{fill_value}'."
                    ),
                    UserWarning,
                    stacklevel=0,
                )
            df[feature] = df[feature].where(~unknown_mask, fill_value)
            df[feature] = df[feature].astype("category").cat.set_categories(categories)
        return df

    def _fillna_as_str(self, series: pd.Series, fill_value: str) -> pd.Series:
        """Fill NaNs with fill_value and cast to string dtype.

        Adds fill_value as a category first when series is already categorical,
        since pandas forbids filling a Categorical with a value outside its
        existing categories.

        Args:
            series (pd.Series): Series to fill, possibly of categorical dtype.
            fill_value (str): Value to replace NaNs with.

        Returns:
            pd.Series: String-dtype series with NaNs replaced by fill_value.
        """
        if (
            isinstance(series.dtype, pd.CategoricalDtype)
            and fill_value not in series.cat.categories
        ):
            series = series.cat.add_categories([fill_value])
        return series.fillna(fill_value).astype("str")

    def _get_fill_value(self, feature: str) -> str:
        """Get the fill value for a feature.

        Args:
            feature (str): Feature name.

        Returns:
            str: Fill value to use for unknown categories.
        """
        fill_values = self.fill_values or {}
        if feature in fill_values:
            return fill_values[feature]
        if self.fill_strategy == "mode":
            return getattr(self, "modes_", {}).get(feature, self.default_value)
        return self.default_value


if __name__ == "__main__":
    df_train = pd.DataFrame(
        {
            "color": pd.Categorical(["red", "blue", "green", "red"]),
            "fuel": pd.Categorical(["petrol", "diesel", "electric", "diesel"]),
            "mileage": [10000, 15000, 12000, 18000],
        }
    )
    df_test = pd.DataFrame(
        {
            "color": ["blue", "purple", "yellow"],
            "fuel": ["diesel", "hydrogen", "petrol"],
            "mileage": [13000, 17000, 9000],
        }
    )

    categorical_aligner = CategoricalAligner(fill_values={"color": "Unknown"})
    categorical_aligner.fit(df_train)

    print("--- test transform ---")
    transformed_test = categorical_aligner.transform(df_test)
    print(transformed_test)
    print(transformed_test["color"].unique())

    print("\n--- train transform (fill category present even without unknowns) ---")
    transformed_train = categorical_aligner.transform(df_train)
    print(transformed_train)
    print(transformed_train["color"].unique())
