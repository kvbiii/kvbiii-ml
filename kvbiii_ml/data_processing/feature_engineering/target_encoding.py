"""Module for target encoding feature generation."""

import gc

import numpy as np
import pandas as pd
from sklearn.model_selection import BaseCrossValidator, KFold
from tqdm import tqdm


class TargetEncodingFeatureGenerator:
    """Class for generating target encoded features."""

    def __init__(
        self,
        features_names: list[str],
        aggregation: str = "mean",
        smooth: int = 10,
        cv: BaseCrossValidator | None = KFold(
            n_splits=5, shuffle=True, random_state=17
        ),
        n_bins: int = 10,
        chunk_size: int = 50000,
        batch_size: int = 10,
        is_continuous_target: bool = False,
        target_bins: int = 4,
    ) -> None:
        """
        Initialize the TargetEncodingFeatureGenerator.

        Args:
            features_names (list[str]): List of feature names to be target encoded.
            aggregation (str, optional): Aggregation method ('mean', 'median',
                'nunique'). Defaults to "mean".
            smooth (int, optional): Smoothing factor to prevent overfitting.
                Defaults to 10.
            cv (BaseCrossValidator | None, optional): Cross-validator for OOF
                encoding. Defaults to KFold(...).
            n_bins (int, optional): Number of bins for float features. Defaults to 10.
            chunk_size (int, optional): Number of rows to process at once during
                transform. Defaults to 50000.
            batch_size (int, optional): Number of features to process at once.
                Defaults to 10.
            is_continuous_target (bool, optional): Whether target is continuous and
                should be binned into quantiles. Defaults to False.
            target_bins (int, optional): Number of quantile bins to divide continuous
                target into. Defaults to 4.
        """
        self.features_names = features_names
        self.aggregation = aggregation
        self.smooth = smooth
        self.cv = cv
        self.n_bins = n_bins
        self.chunk_size = chunk_size
        self.batch_size = batch_size
        self.is_continuous_target = is_continuous_target
        self.target_bins = target_bins
        self.bin_edges_: dict[str, np.ndarray] = {}
        self.group_stats_: dict[str, pd.Series] = {}
        self.global_stat: float = 0.0
        self.target_bin_edges_: np.ndarray | None = None
        self._fitted_index: pd.Index | None = None
        self._fitted_te_df: pd.DataFrame | None = None
        self._validate_init_params()

    def _validate_init_params(self) -> None:
        """
        Validate the initialization parameters.

        Raises:
            ValueError: If parameters are invalid.
        """
        if not isinstance(self.features_names, list) or not all(
            isinstance(name, str) for name in self.features_names
        ):
            raise ValueError("features_names must be a list of strings.")
        if self.aggregation not in ["mean", "median", "nunique"]:
            raise ValueError(
                f"Unsupported aggregation method: {self.aggregation}. "
                "Supported methods are 'mean', 'median', 'nunique'."
            )
        if not isinstance(self.smooth, int) or self.smooth < 0:
            raise ValueError("smooth must be a non-negative integer.")
        if self.cv is not None and not isinstance(self.cv, BaseCrossValidator):
            raise ValueError("cv must be None or an instance of BaseCrossValidator.")
        if not isinstance(self.n_bins, int) or self.n_bins < 2:
            raise ValueError("n_bins must be an integer >= 2.")
        if not isinstance(self.chunk_size, int) or self.chunk_size < 1:
            raise ValueError("chunk_size must be a positive integer.")
        if not isinstance(self.batch_size, int) or self.batch_size < 1:
            raise ValueError("batch_size must be a positive integer.")
        if not isinstance(self.is_continuous_target, bool):
            raise ValueError("is_continuous_target must be a boolean.")
        if not isinstance(self.target_bins, int) or self.target_bins < 2:
            raise ValueError("target_bins must be an integer >= 2.")

    def _bin_continuous_target(self, y: pd.Series) -> pd.Series:
        """
        Bin a continuous target into quantile-based discrete categories.

        Args:
            y (pd.Series): Continuous target series.

        Returns:
            pd.Series: Binned target as a Series of integers.
        """
        if self.target_bin_edges_ is None:
            self.target_bin_edges_ = np.quantile(
                y, np.linspace(0, 1, self.target_bins + 1)
            )
            self.target_bin_edges_[-1] *= 1.001

        labels = list(range(self.target_bins))
        return pd.cut(
            y, bins=self.target_bin_edges_, labels=labels, include_lowest=True
        ).astype(int)

    def _prepare_target(self, y: pd.Series) -> pd.Series:
        """
        Prepare target for encoding by binning if continuous.

        Args:
            y (pd.Series): Target series.

        Returns:
            pd.Series: Prepared target series (binned if continuous).
        """
        if self.is_continuous_target:
            return self._bin_continuous_target(y)
        return y

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
            pd.Series: Binned feature as a Series of strings.
        """
        if feature_name in self.bin_edges_:
            bin_edges = self.bin_edges_[feature_name]
        else:
            bin_edges = np.linspace(
                X[feature_name].min(), X[feature_name].max(), self.n_bins + 1
            )
            bin_edges[-1] *= 1.001
            self.bin_edges_[feature_name] = bin_edges

        labels = [f"bin_{i}" for i in range(len(bin_edges) - 1)]
        return pd.cut(
            X[feature_name], bins=bin_edges, labels=labels, include_lowest=True
        ).astype(str)

    def _compute_feature_stats(
        self, X: pd.DataFrame, y: pd.Series, feature_name: str
    ) -> pd.DataFrame:
        """
        Compute statistics for the target variable grouped by a feature.

        Args:
            X (pd.DataFrame): Feature DataFrame.
            y (pd.Series): Target series.
            feature_name (str): Name of the feature to compute statistics for.

        Returns:
            pd.DataFrame: DataFrame with computed statistics.
        """
        prepared_target = self._prepare_target(y)

        feature_values = (
            self._bin_float_feature(X, feature_name)
            if self._is_float_feature(X, feature_name)
            else X[feature_name]
        )

        group_stats = (
            prepared_target.groupby(
                feature_values.rename(feature_name), sort=False, observed=True
            )
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

    def _encode_feature_chunk(
        self, chunk: pd.DataFrame, feature_name: str, mapping: pd.Series
    ) -> np.ndarray:
        """
        Encode a chunk of data for a given feature.

        Args:
            chunk (pd.DataFrame): DataFrame chunk to encode.
            feature_name (str): Name of the feature to encode.
            mapping (pd.Series): Mapping from feature values to encoded values.

        Returns:
            np.ndarray: Array of encoded values.
        """
        feature_series = (
            self._bin_float_feature(chunk, feature_name)
            if self._is_float_feature(chunk, feature_name)
            else chunk[feature_name]
        )
        return (
            feature_series.map(mapping).fillna(self.global_stat).astype("float32")
        ).values

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "TargetEncodingFeatureGenerator":
        """
        Fit the target encoder on the provided data.

        Args:
            X (pd.DataFrame): Feature DataFrame.
            y (pd.Series): Target series.

        Returns:
            TargetEncodingFeatureGenerator: The fitted encoder instance.
        """
        if not self.features_names:
            self.features_names = list(X.columns)
        self._validate_fit_inputs(X, y)

        prepared_target = self._prepare_target(y)
        self.global_stat = float(getattr(prepared_target, self.aggregation)())
        self.group_stats_ = {}
        self.bin_edges_ = {}

        n_rows = len(X)
        te_columns_dict: dict[str, np.ndarray] = {}

        pbar = tqdm(total=len(self.features_names), desc="Fitting target encoding")

        for feat_idx in range(0, len(self.features_names), self.batch_size):
            batch_features = self.features_names[feat_idx : feat_idx + self.batch_size]

            for feature_name in batch_features:
                te_col_name = f"TE_{self.aggregation.upper()}_{feature_name}"

                if self.cv:
                    te_values = np.empty(n_rows, dtype=np.float32)
                    for train_idx, valid_idx in self.cv.split(X, y):
                        X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
                        X_valid = X.iloc[valid_idx]

                        fold_stats = self._compute_feature_stats(
                            X_train, y_train, feature_name
                        )
                        mapping = fold_stats.set_index(feature_name)["te_value"]
                        te_values[valid_idx] = self._encode_feature_chunk(
                            X_valid, feature_name, mapping
                        )
                        del fold_stats, mapping

                    te_columns_dict[te_col_name] = te_values
                    gc.collect()

                full_stats = self._compute_feature_stats(X, y, feature_name)
                self.group_stats_[feature_name] = full_stats.set_index(feature_name)[
                    "te_value"
                ]
                del full_stats

                if not self.cv:
                    te_columns_dict[te_col_name] = self._encode_feature_chunk(
                        X, feature_name, self.group_stats_[feature_name]
                    )

            pbar.update(len(batch_features))
            gc.collect()

        pbar.close()

        self._fitted_index = X.index.copy()
        self._fitted_te_df = pd.DataFrame(te_columns_dict, index=X.index)
        del te_columns_dict
        gc.collect()

        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Transform the data using learned target encodings.

        Args:
            X (pd.DataFrame): Feature DataFrame to transform.

        Returns:
            pd.DataFrame: DataFrame with target-encoded features.

        Raises:
            ValueError: If the encoder has not been fitted yet.
        """
        if not self.group_stats_:
            raise ValueError(
                "TargetEncodingFeatureGenerator must be fitted before transform."
            )

        if (
            self._fitted_index is not None
            and self._fitted_te_df is not None
            and X.index.equals(self._fitted_index)
        ):
            return pd.concat([X, self._fitted_te_df], axis=1)

        n_rows = len(X)
        n_chunks = (n_rows + self.chunk_size - 1) // self.chunk_size

        te_columns_dict: dict[str, np.ndarray] = {
            f"TE_{self.aggregation.upper()}_{feature_name}": np.empty(
                n_rows, dtype=np.float32
            )
            for feature_name in self.features_names
        }

        total_features = len(self.features_names)
        pbar = tqdm(total=total_features, desc="Transforming with target encoding")

        for feat_idx in range(0, total_features, self.batch_size):
            batch_features = self.features_names[feat_idx : feat_idx + self.batch_size]

            for chunk_idx in range(n_chunks):
                start_idx = chunk_idx * self.chunk_size
                end_idx = min((chunk_idx + 1) * self.chunk_size, n_rows)
                chunk = X.iloc[start_idx:end_idx]

                for feature_name in batch_features:
                    te_col_name = f"TE_{self.aggregation.upper()}_{feature_name}"
                    mapping = self.group_stats_[feature_name]
                    result = self._encode_feature_chunk(chunk, feature_name, mapping)
                    te_columns_dict[te_col_name][start_idx:end_idx] = result
                    del result

                del chunk
                gc.collect()

            pbar.update(len(batch_features))

        pbar.close()

        te_columns_df = pd.DataFrame(te_columns_dict, index=X.index)
        result_df = pd.concat([X, te_columns_df], axis=1)

        del te_columns_dict, te_columns_df
        gc.collect()

        return result_df

    def fit_transform(self, X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
        """
        Fit the encoder and transform the data.

        Args:
            X (pd.DataFrame): Feature DataFrame.
            y (pd.Series): Target series.

        Returns:
            pd.DataFrame: DataFrame with target-encoded features.
        """
        return self.fit(X, y).transform(X)

    def get_feature_stats(
        self, X: pd.DataFrame, y: pd.Series, feature_name: str
    ) -> pd.DataFrame:
        """
        Compute statistics for the target variable grouped by a feature.

        Args:
            X (pd.DataFrame): Feature DataFrame.
            y (pd.Series): Target series.
            feature_name (str): Name of the feature to compute statistics for.

        Returns:
            pd.DataFrame: DataFrame with computed statistics.
        """
        return self._compute_feature_stats(X, y, feature_name)

    def _validate_fit_inputs(self, X: pd.DataFrame, y: pd.Series) -> None:
        """
        Validate inputs for fit and fit_transform.

        Args:
            X (pd.DataFrame): Feature DataFrame.
            y (pd.Series): Target series.

        Raises:
            TypeError: If X is not a DataFrame or y is not array-like.
            ValueError: If X and y have different lengths.
            KeyError: If features are missing from X.
        """
        if not isinstance(X, pd.DataFrame):
            raise TypeError("X must be a pandas DataFrame.")
        if not isinstance(y, (pd.Series, pd.DataFrame, np.ndarray, list, tuple)):
            raise TypeError("y must be a pandas Series or array-like.")
        y_len = len(y)
        if len(X) != y_len:
            raise ValueError(
                f"X and y must have the same number of rows. "
                f"Got len(X)={len(X)} and len(y)={y_len}."
            )
        missing = [f for f in self.features_names if f not in X.columns]
        if missing:
            raise KeyError(f"Missing features in X: {missing}")


if __name__ == "__main__":
    data = {
        "feature1": ["A", "B", "A", "C", "B"],
        "feature2": ["X", "Y", "X", "Z", "Y"],
        "num_feature": [10, 20, 10, 30, 20],
        "target": [1, 0, 1, 0, 1],
    }
    df = pd.DataFrame(data)

    te_generator = TargetEncodingFeatureGenerator(
        features_names=["feature1", "feature2", "num_feature"],
        aggregation="mean",
        smooth=10,
        cv=None,
    )

    transformed_df = te_generator.fit_transform(
        df[["feature1", "feature2", "num_feature"]], df["target"]
    )
    print(transformed_df)
    for feature in te_generator.features_names:
        print(f"\nGroup stats for {feature}:")
        print(te_generator.group_stats_[feature])

    print("\n--- Testing continuous target ---")
    data_continuous = {
        "feature1": ["A", "B", "A", "C", "B", "A", "C", "B"],
        "feature2": ["X", "Y", "X", "Z", "Y", "Y", "X", "Z"],
        "num_feature": [10, 20, 10, 30, 20, 15, 25, 18],
        "target": [1.5, 2.3, 1.8, 4.5, 3.1, 2.0, 4.2, 3.8],
    }
    df_continuous = pd.DataFrame(data_continuous)

    te_generator_continuous = TargetEncodingFeatureGenerator(
        features_names=["feature1", "feature2", "num_feature"],
        aggregation="mean",
        smooth=5,
        cv=None,
        is_continuous_target=True,
        target_bins=4,
    )

    transformed_df_continuous = te_generator_continuous.fit_transform(
        df_continuous[["feature1", "feature2", "num_feature"]], df_continuous["target"]
    )
    print(transformed_df_continuous)
    for feature in te_generator_continuous.features_names:
        print(f"\nGroup stats for {feature}:")
        print(te_generator_continuous.group_stats_[feature])
