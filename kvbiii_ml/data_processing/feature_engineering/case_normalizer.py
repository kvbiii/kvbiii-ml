import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class CaseNormalizer(BaseEstimator, TransformerMixin):
    """
    Transformer that normalizes case for object, string, or category features.
    """

    _NORMALIZATIONS = {"lower", "upper", "title", "capitalize", "casefold"}

    def __init__(
        self,
        features_names: list[str] | None = None,
        normalization: str = "lower",
    ) -> None:
        """
        Initialize the CaseNormalizer.

        Args:
            features_names (list[str] | None, optional): Feature names to normalize.
                If None, object, string, and category columns are auto-detected.
            normalization (str, optional): Case normalization strategy. Supported values
                are: "lower", "upper", "title", "capitalize", and "casefold".
        """
        self.features_names = features_names
        self.normalization = normalization
        self.features_names_: list[str] = []
        self.input_features_: pd.Index = pd.Index([])
        self._is_fitted = False
        self._validate_init_params()

    def _validate_init_params(self) -> None:
        """
        Validate the initialization parameters.

        Raises:
            ValueError: If parameters are invalid.
        """
        if self.features_names is not None and (
            not isinstance(self.features_names, list)
            or not all(isinstance(name, str) for name in self.features_names)
        ):
            raise ValueError("features_names must be None or a list of strings.")
        if self.normalization not in self._NORMALIZATIONS:
            raise ValueError(
                "normalization must be one of: "
                + ", ".join(sorted(self._NORMALIZATIONS))
                + "."
            )

    def fit(
        self,
        df: pd.DataFrame,
        _y: pd.Series | None = None,
    ) -> "CaseNormalizer":
        """
        Fit the normalizer by determining features to normalize.

        Args:
            df (pd.DataFrame): Training data.
            _y (pd.Series | None, optional): Target (unused). Defaults to None.

        Returns:
            CaseNormalizer: Fitted transformer.
        """
        self.input_features_ = df.columns
        if self.features_names is None:
            self.features_names_ = df.select_dtypes(
                include=["object", "string", "category"]
            ).columns.tolist()
        else:
            self.features_names_ = [
                name for name in self.features_names if name in df.columns
            ]
        self._is_fitted = True
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Transform data by normalizing case for configured features.

        Args:
            df (pd.DataFrame): Data to transform.

        Returns:
            pd.DataFrame: Transformed data with normalized case.

        Raises:
            ValueError: If the transformer has not been fitted.
        """
        if not self._is_fitted:
            raise ValueError("CaseNormalizer must be fitted before transform.")

        df = df.copy()
        for feature in self.features_names_:
            if feature not in df.columns:
                continue
            series = df[feature]
            if pd.api.types.is_string_dtype(series.dtype):
                df[feature] = self._normalize_string_series(series.astype("string"))
            elif pd.api.types.is_object_dtype(series.dtype):
                df[feature] = self._normalize_object_series(series)
        return df

    def get_feature_names_out(
        self,
        _input_features: list[str] | None = None,
    ) -> pd.Index:
        """
        Return feature names seen during fit.

        Args:
            _input_features (list[str] | None, optional): Unused. For API consistency.

        Returns:
            pd.Index: Original feature names.
        """
        return self.input_features_

    def _normalize_string_series(self, series: pd.Series) -> pd.Series:
        """
        Normalize case for pandas string series.

        Args:
            series (pd.Series): String series to normalize.

        Returns:
            pd.Series: Normalized string series.
        """
        if self.normalization == "lower":
            return series.str.lower()
        if self.normalization == "upper":
            return series.str.upper()
        if self.normalization == "title":
            return series.str.title()
        if self.normalization == "capitalize":
            return series.str.capitalize()
        return series.str.casefold()

    def _normalize_object_series(self, series: pd.Series) -> pd.Series:
        """
        Normalize case for object-like series while preserving non-strings.

        Args:
            series (pd.Series): Series to normalize.

        Returns:
            pd.Series: Normalized series.
        """
        return series.map(self._normalize_value)

    def _normalize_value(self, value: object) -> object:
        """
        Normalize case for a single value when it is a string.

        Args:
            value (object): Value to normalize.

        Returns:
            object: Normalized value or original if not a string.
        """
        if pd.isna(value):
            return value
        if not isinstance(value, str):
            return value
        if self.normalization == "lower":
            return value.lower()
        if self.normalization == "upper":
            return value.upper()
        if self.normalization == "title":
            return value.title()
        if self.normalization == "capitalize":
            return value.capitalize()
        return value.casefold()


if __name__ == "__main__":
    data = pd.DataFrame(
        {
            "obj": ["A", "b", None],
            "str": pd.Series(["X", "y", pd.NA], dtype="string"),
            "num": [1, 2, 3],
        }
    )
    normalizer = CaseNormalizer(features_names=["obj", "str"], normalization="lower")
    normalizer.fit(data)
    transformed = normalizer.transform(data)
    print(transformed)
