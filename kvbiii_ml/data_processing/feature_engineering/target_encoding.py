import pandas as pd
import numpy as np
from sklearn.model_selection import BaseCrossValidator, KFold
from tqdm import tqdm


class TargetEncodingFeatureGenerator:
    """
    Class for generating target encoded features.
    """

    def __init__(
        self,
        features_names: list[str],
        aggregation: str = "mean",
        smooth: int = 10,
        cv: BaseCrossValidator | None = KFold(
            n_splits=5, shuffle=True, random_state=17
        ),
        n_bins: int = 10,
    ) -> None:
        """
        Initialize the TargetEncodingFeatureGenerator.

        Args:
            features_names (list[str]): List of feature names to be target encoded.
            aggregation (str): Aggregation method for target encoding ('mean', 'median', 'nunique').
            smooth (int): Smoothing factor to prevent overfitting.
            cv (BaseCrossValidator): Cross-validator for ensuring no data leakage during encoding.
            n_bins (int): Number of bins to use for float features.
        """
        self.features_names = features_names
        self.aggregation = aggregation
        self.smooth = smooth
        self.cv = cv
        self.n_bins = n_bins
        self.bin_edges_ = {}  # Store bin edges for float features
        self._validate_init_params()

    def _validate_init_params(self) -> None:
        """
        Validate the parameters for target encoding.
        """
        if not isinstance(self.features_names, list) or not all(
            isinstance(name, str) for name in self.features_names
        ):
            raise ValueError("features_names must be a list of strings.")
        if self.aggregation not in ["mean", "median", "nunique"]:
            raise ValueError(
                f"Unsupported aggregation method: {self.aggregation}. Supported methods are 'mean', 'median', 'nunique'."
            )
        if not isinstance(self.smooth, int) or self.smooth < 0:
            raise ValueError("smooth must be a non-negative integer.")
        if self.cv is not None and not isinstance(self.cv, BaseCrossValidator):
            raise ValueError(
                "cv must be None or an instance of BaseCrossValidator or its subclasses."
            )
        if not isinstance(self.n_bins, int) or self.n_bins < 2:
            raise ValueError("n_bins must be an integer >= 2.")

    def _is_float_feature(self, X: pd.DataFrame, feature_name: str) -> bool:
        """
        Check if a feature has float dtype.

        Args:
            X (pd.DataFrame): Feature DataFrame.
            feature_name (str): Name of the feature to check.

        Returns:
            bool: True if the feature has float dtype, False otherwise.
        """
        return pd.api.types.is_float_dtype(X[feature_name])

    def _bin_float_feature(self, X: pd.DataFrame, feature_name: str) -> pd.Series:
        """
        Bin a float feature into discrete categories.

        Args:
            X (pd.DataFrame): Feature DataFrame.
            feature_name (str): Name of the float feature to bin.

        Returns:
            pd.Series: Binned feature.
        """
        if feature_name in self.bin_edges_:
            # Use stored bin edges for consistent binning
            bin_edges = self.bin_edges_[feature_name]
            labels = [f"bin_{i}" for i in range(len(bin_edges) - 1)]
            return pd.cut(
                X[feature_name], bins=bin_edges, labels=labels, include_lowest=True
            )
        else:
            # Calculate bin edges and store them
            bin_edges = np.linspace(
                X[feature_name].min(), X[feature_name].max(), self.n_bins + 1
            )
            # Ensure the max value is included
            bin_edges[-1] = bin_edges[-1] * 1.001

            self.bin_edges_[feature_name] = bin_edges
            labels = [f"bin_{i}" for i in range(len(bin_edges) - 1)]
            return pd.cut(
                X[feature_name], bins=bin_edges, labels=labels, include_lowest=True
            )

    def fit_transform(self, X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
        """
        Fit and transform the DataFrame with target encoding.

        Args:
            X (pd.DataFrame): Feature DataFrame containing columns listed in features_names.
            y (pd.Series): Target series aligned with X.

        Returns:
            pd.DataFrame: DataFrame with the target-encoded feature added.
        """
        # If no features specified, use all columns from X
        if not self.features_names:
            self.features_names = list(X.columns)
        # Validate inputs
        self._validate_fit_inputs(X, y)
        self.global_stat = getattr(y, self.aggregation)()
        self.group_stats_: dict[str, pd.Series] = {}
        te_columns: dict[str, np.ndarray] = {}

        for feature_name in tqdm(self.features_names, desc="Computing target encoding"):
            te_col_name = f"TE_{self.aggregation.upper()}_{feature_name}"
            if self.cv is not None:
                # Out-of-fold target encoding using the provided cross-validator
                te_values = np.zeros(len(X), dtype=np.float32)
                # Use CV splits (no target leakage)
                for train_idx, valid_idx in self.cv.split(X, y):
                    fold_stats = self.get_feature_stats(
                        X.iloc[train_idx], y.iloc[train_idx], feature_name
                    )
                    mapping = fold_stats.set_index(feature_name)["te_value"]
                    vals = (
                        X.iloc[valid_idx][feature_name]
                        .map(mapping)
                        .fillna(self.global_stat)
                        .astype("float32")
                        .values
                    )
                    te_values[valid_idx] = vals
                te_columns[te_col_name] = te_values
                full_stats = self.get_feature_stats(X, y, feature_name)
                self.group_stats_[feature_name] = full_stats.set_index(feature_name)[
                    "te_value"
                ]
            else:
                stats = self.get_feature_stats(X, y, feature_name)
                mapping = stats.set_index(feature_name)["te_value"]
                te_values = (
                    X[feature_name]
                    .map(mapping)
                    .fillna(self.global_stat)
                    .astype("float32")
                    .values
                )
                te_columns[te_col_name] = te_values
                self.group_stats_[feature_name] = mapping

        te_columns_df = pd.DataFrame(te_columns, index=X.index)
        X_encoded = pd.concat([X, te_columns_df], axis=1)
        return X_encoded

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "TargetEncodingFeatureGenerator":
        """
        Fit the target encoder on the provided features X and target y.

        Args:
            X (pd.DataFrame): Feature DataFrame containing columns listed in features_names.
            y (pd.Series): Target series aligned with X.

        Returns:
            self: Fitted instance with learned group statistics.
        """
        # If no features specified, use all columns from X
        if not self.features_names:
            self.features_names = list(X.columns)
        # Validate inputs
        self._validate_fit_inputs(X, y)
        # Global statistic of the target for smoothing / fallback
        self.global_stat = getattr(y, self.aggregation)()
        self.group_stats_: dict[str, pd.Series] = {}
        self.bin_edges_ = {}  # Reset bin edges

        # Compute and store training encodings (OOF if cv is provided)
        te_columns: dict[str, np.ndarray] = {}
        for feature_name in tqdm(self.features_names, desc="Computing target encoding"):
            te_col_name = f"TE_{self.aggregation.upper()}_{feature_name}"
            if self.cv is not None:
                te_values = np.zeros(len(X), dtype=np.float32)
                # Perform out-of-fold target encoding to prevent target leakage
                for train_idx, valid_idx in self.cv.split(X, y):
                    fold_stats = self.get_feature_stats(
                        X.iloc[train_idx], y.iloc[train_idx], feature_name
                    )
                    mapping = fold_stats.set_index(feature_name)["te_value"]

                    # Apply mapping based on whether it's a float feature or not
                    if self._is_float_feature(X, feature_name):
                        # For float features, bin the validation data using training bins
                        binned_valid_values = self._bin_float_feature(
                            X.iloc[valid_idx], feature_name
                        )
                        vals = (
                            binned_valid_values.astype(str)
                            .map(mapping)
                            .fillna(self.global_stat)
                            .astype("float32")
                            .values
                        )
                    else:
                        vals = (
                            X.iloc[valid_idx][feature_name]
                            .map(mapping)
                            .fillna(self.global_stat)
                            .astype("float32")
                            .values
                        )
                    te_values[valid_idx] = vals

                te_columns[te_col_name] = te_values

                # Compute stats on full dataset for transform
                full_stats = self.get_feature_stats(X, y, feature_name)
                self.group_stats_[feature_name] = full_stats.set_index(feature_name)[
                    "te_value"
                ]
            else:
                stats = self.get_feature_stats(X, y, feature_name)
                mapping = stats.set_index(feature_name)["te_value"]

                if self._is_float_feature(X, feature_name):
                    # For float features, bin the data before mapping
                    binned_values = self._bin_float_feature(X, feature_name)
                    te_values = (
                        binned_values.map(mapping)
                        .fillna(self.global_stat)
                        .astype("float32")
                        .values
                    )
                else:
                    te_values = (
                        X[feature_name]
                        .map(mapping)
                        .fillna(self.global_stat)
                        .astype("float32")
                        .values
                    )

                te_columns[te_col_name] = te_values
                self.group_stats_[feature_name] = mapping

        # Store fitted encodings for reuse on the same data in transform()
        self._fitted_index = X.index.copy()
        self._fitted_te_df = pd.DataFrame(te_columns, index=X.index)

        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Transform the DataFrame with target encoding.

        Args:
            X (pd.DataFrame): Feature DataFrame to be transformed.

        Returns:
            pd.DataFrame: DataFrame with the target-encoded feature added.
        """
        # If called on the same data as fit (by index), reuse stored encodings (OOF if cv was used)
        if hasattr(self, "_fitted_index") and X.index.equals(self._fitted_index):
            te_columns_df = self._fitted_te_df.reindex(X.index)
            return pd.concat([X, te_columns_df], axis=1)

        # Otherwise, encode using learned mappings
        te_columns: dict[str, np.ndarray] = {}
        for feature_name in tqdm(
            self.features_names, desc="Transforming with target encoding"
        ):
            te_col_name = f"TE_{self.aggregation.upper()}_{feature_name}"
            mapping = self.group_stats_[feature_name]

            if self._is_float_feature(X, feature_name):
                # For float features, bin the data before mapping
                binned_values = self._bin_float_feature(X, feature_name)
                te_columns[te_col_name] = (
                    binned_values.astype(str)
                    .map(mapping)
                    .fillna(self.global_stat)
                    .astype("float32")
                    .values
                )
            else:
                te_columns[te_col_name] = (
                    X[feature_name]
                    .map(mapping)
                    .fillna(self.global_stat)
                    .astype("float32")
                    .values
                )

        te_columns_df = pd.DataFrame(te_columns, index=X.index)
        return pd.concat([X, te_columns_df], axis=1)

    def get_feature_stats(
        self, X: pd.DataFrame, y: pd.Series, feature_name: str
    ) -> pd.DataFrame:
        """
        Compute statistics for the target variable grouped by a feature.

        Args:
            X (pd.DataFrame): Feature DataFrame containing the feature.
            y (pd.Series): Target series aligned with X.
            feature_name (str): Name of the feature for which to compute statistics.

        Returns:
            pd.DataFrame: DataFrame with the computed statistics.
        """
        # Check if feature is float type and bin it if necessary
        if self._is_float_feature(X, feature_name):
            feature_values = self._bin_float_feature(X, feature_name)
        else:
            feature_values = X[feature_name].rename(feature_name)

        group_stats = (
            y.groupby(feature_values, sort=False, observed=True)
            .agg([self.aggregation, "count"])
            .reset_index()
        )

        if self.aggregation == "nunique":
            group_stats["te_value"] = (
                group_stats[self.aggregation] / group_stats["count"]
            ).astype("float32")
        else:
            group_stats["te_value"] = (
                (group_stats[self.aggregation] * group_stats["count"])
                + (self.global_stat * self.smooth)
            ) / (group_stats["count"] + self.smooth)
            group_stats["te_value"] = group_stats["te_value"].astype("float32")
        return group_stats

    def _validate_fit_inputs(self, X: pd.DataFrame, y: pd.Series) -> None:
        """
        Validate X and y before fitting/transforming.

        Ensures:
        - X is a DataFrame and y is a Series or array-like convertible to Series
        - X and y have the same length
        - All features_names exist in X columns

        Args:
            X (pd.DataFrame): Feature DataFrame.
            y (pd.Series): Target series.
        """
        if not isinstance(X, pd.DataFrame):
            raise TypeError("X must be a pandas DataFrame.")
        if not isinstance(y, (pd.Series, pd.DataFrame, np.ndarray, list, tuple)):
            raise TypeError("y must be a pandas Series or array-like.")
        y_len = len(y)
        if len(X) != y_len:
            raise ValueError(
                f"X and y must have the same number of rows. Got len(X)={len(X)} and len(y)={y_len}."
            )
        missing = [f for f in self.features_names if f not in X.columns]
        if missing:
            raise KeyError(f"Missing features in X: {missing}")


if __name__ == "__main__":
    # Example usage
    import pandas as pd

    # Sample DataFrame
    data = {
        "feature1": ["A", "B", "A", "C", "B"],
        "feature2": ["X", "Y", "X", "Z", "Y"],
        "num_feature": [10, 20, 10, 30, 20],
        "target": [1, 0, 1, 0, 1],
    }
    df = pd.DataFrame(data)

    # Initialize the TargetEncodingFeatureGenerator (no target_name in __init__)
    te_generator = TargetEncodingFeatureGenerator(
        features_names=["feature1", "feature2", "num_feature"],
        aggregation="mean",
        smooth=10,
        cv=None,
    )

    # Fit and transform using X and y
    transformed_df = te_generator.fit_transform(
        df[["feature1", "feature2", "num_feature"]], df["target"]
    )
    print(transformed_df)
    for feature in te_generator.features_names:
        print(f"\nGroup stats for {feature}:")
        print(te_generator.group_stats_[feature])
