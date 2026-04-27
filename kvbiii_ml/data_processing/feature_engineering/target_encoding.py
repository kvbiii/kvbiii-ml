import gc
from typing import Any

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
        config: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        """
        Initialize the TargetEncodingFeatureGenerator.

        Args:
            features_names (list[str]): List of feature names to be target encoded.
            aggregation (str, optional): Aggregation method ('mean', 'median',
                'nunique'). Defaults to "mean".
            config (dict[str, Any] | None, optional): Runtime configuration values.
                Supported keys: smooth, cv, n_bins, chunk_size, batch_size,
                is_continuous_target, target_bins, random_state.
            **kwargs (Any): Backward-compatible config values passed directly as
                keyword arguments. These override values from config.
        """
        default_config: dict[str, Any] = {
            "smooth": 10,
            "cv": KFold(n_splits=5, shuffle=True, random_state=17),
            "n_bins": 10,
            "chunk_size": 50000,
            "batch_size": 10,
            "is_continuous_target": False,
            "target_bins": 4,
            "random_state": 17,
        }
        merged_config = dict(default_config)
        if config:
            merged_config.update(config)
        if kwargs:
            merged_config.update(kwargs)

        self.features_names = features_names
        self.aggregation = aggregation
        self.config = merged_config
        self.bin_edges_: dict[str, np.ndarray] = {}
        self.group_stats_: dict[str, pd.Series] = {}
        self.global_stat: float = 0.0
        self._state: dict[str, Any] = {
            "target_bin_edges": None,
            "fitted_index": None,
            "fitted_te_df": None,
        }
        self._validate_init_params()

        if self.random_state is not None:
            np.random.seed(self.random_state)

    @property
    def smooth(self) -> int:
        """Return smoothing factor used in target encoding."""
        return int(self.config["smooth"])

    @property
    def cv(self) -> BaseCrossValidator | None:
        """Return cross-validator used for out-of-fold encoding."""
        return self.config["cv"]

    @property
    def n_bins(self) -> int:
        """Return number of bins for float features."""
        return int(self.config["n_bins"])

    @property
    def chunk_size(self) -> int:
        """Return transform chunk size."""
        return int(self.config["chunk_size"])

    @property
    def batch_size(self) -> int:
        """Return batch size setting for compatibility."""
        return int(self.config["batch_size"])

    @property
    def is_continuous_target(self) -> bool:
        """Return whether the target should be binned."""
        return bool(self.config["is_continuous_target"])

    @property
    def target_bins(self) -> int:
        """Return number of quantile bins for continuous targets."""
        return int(self.config["target_bins"])

    @property
    def random_state(self) -> int | None:
        """Return random state used for deterministic behavior."""
        value = self.config["random_state"]
        return int(value) if value is not None else None

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
        if self.random_state is not None and not isinstance(self.random_state, int):
            raise ValueError("random_state must be None or an integer.")

    def _bin_continuous_target(self, y: pd.Series) -> pd.Series:
        """
        Bin a continuous target into quantile-based discrete categories.

        Args:
            y (pd.Series): Continuous target series.

        Returns:
            pd.Series: Binned target as a Series of integers.
        """
        target_bin_edges = self._state["target_bin_edges"]
        if target_bin_edges is None:
            quantiles = np.linspace(0, 1, self.target_bins + 1)
            target_bin_edges = np.quantile(y, quantiles)
            target_bin_edges = np.round(target_bin_edges, decimals=10)
            target_bin_edges[-1] = target_bin_edges[-1] * 1.001
            self._state["target_bin_edges"] = target_bin_edges

        labels = list(range(self.target_bins))
        return pd.cut(
            y, bins=target_bin_edges, labels=labels, include_lowest=True
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

    def _is_float_feature(self, df: pd.DataFrame, feature_name: str) -> bool:
        """
        Check if a feature has float dtype.

        Args:
            df (pd.DataFrame): Feature DataFrame.
            feature_name (str): Name of the feature to check.

        Returns:
            bool: True if the feature has float dtype, False otherwise.
        """
        return pd.api.types.is_float_dtype(df[feature_name])

    def _bin_float_feature(self, df: pd.DataFrame, feature_name: str) -> pd.Series:
        """
        Bin a float feature into discrete categories.

        Args:
            df (pd.DataFrame): Feature DataFrame.
            feature_name (str): Name of the float feature to bin.

        Returns:
            pd.Series: Binned feature as a Series of strings.
        """
        if feature_name in self.bin_edges_:
            bin_edges = self.bin_edges_[feature_name]
        else:
            min_val = df[feature_name].min()
            max_val = df[feature_name].max()
            if min_val == max_val:
                bin_edges = np.array([min_val - 0.001, max_val + 0.001])
            else:
                bin_edges = np.linspace(min_val, max_val, self.n_bins + 1)
                bin_edges = np.round(bin_edges, decimals=10)
                bin_edges[-1] = bin_edges[-1] * 1.001
            self.bin_edges_[feature_name] = bin_edges

        labels = [f"bin_{idx}" for idx in range(len(bin_edges) - 1)]
        return pd.cut(
            df[feature_name], bins=bin_edges, labels=labels, include_lowest=True
        ).astype(str)

    def _compute_feature_stats(
        self, df: pd.DataFrame, y: pd.Series, feature_name: str
    ) -> pd.DataFrame:
        """
        Compute statistics for the target variable grouped by a feature.

        Args:
            df (pd.DataFrame): Feature DataFrame.
            y (pd.Series): Target series.
            feature_name (str): Name of the feature to compute statistics for.

        Returns:
            pd.DataFrame: DataFrame with computed statistics.
        """
        prepared_target = self._prepare_target(y)

        feature_values = (
            self._bin_float_feature(df, feature_name)
            if self._is_float_feature(df, feature_name)
            else df[feature_name]
        )

        group_stats = (
            prepared_target.groupby(
                feature_values.rename(feature_name), sort=True, observed=True
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

    def _fit_single_feature(
        self,
        df: pd.DataFrame,
        prepared_target: pd.Series,
        feature_name: str,
    ) -> tuple[str, np.ndarray]:
        """Fit target encoding artifacts for a single feature."""
        n_rows = len(df)
        te_col_name = f"TE_{self.aggregation.upper()}_{feature_name}"
        te_values = np.empty(n_rows, dtype=np.float32)

        if self.cv is not None:
            for train_idx, valid_idx in self.cv.split(df, prepared_target):
                df_train = df.iloc[train_idx]
                target_train = prepared_target.iloc[train_idx]
                df_valid = df.iloc[valid_idx]

                fold_stats = self._compute_feature_stats(
                    df_train, target_train, feature_name
                )
                mapping = fold_stats.set_index(feature_name)["te_value"]
                te_values[valid_idx] = self._encode_feature_chunk(
                    df_valid, feature_name, mapping
                )
                del fold_stats, mapping
        else:
            full_mapping = self.group_stats_[feature_name]
            te_values = self._encode_feature_chunk(df, feature_name, full_mapping)

        return te_col_name, te_values

    def fit(self, df: pd.DataFrame, y: pd.Series) -> "TargetEncodingFeatureGenerator":
        """
        Fit the target encoder on the provided data.

        Args:
            df (pd.DataFrame): Feature DataFrame.
            y (pd.Series): Target series.

        Returns:
            TargetEncodingFeatureGenerator: The fitted encoder instance.
        """
        if not self.features_names:
            self.features_names = list(df.columns)
        self._validate_fit_inputs(df, y)

        prepared_target = self._prepare_target(y)
        self.global_stat = float(getattr(prepared_target, self.aggregation)())
        self.group_stats_ = {}
        self.bin_edges_ = {}

        te_columns_dict: dict[str, np.ndarray] = {}
        pbar = tqdm(total=len(self.features_names), desc="Fitting target encoding")

        for feature_name in self.features_names:
            full_stats = self._compute_feature_stats(df, prepared_target, feature_name)
            self.group_stats_[feature_name] = full_stats.set_index(feature_name)[
                "te_value"
            ]
            del full_stats

            te_col_name, te_values = self._fit_single_feature(
                df, prepared_target, feature_name
            )
            te_columns_dict[te_col_name] = te_values
            pbar.update(1)
            gc.collect()

        pbar.close()

        self._state["fitted_index"] = df.index.copy()
        self._state["fitted_te_df"] = pd.DataFrame(te_columns_dict, index=df.index)
        del te_columns_dict
        gc.collect()

        return self

    def _transform_single_feature(
        self,
        df: pd.DataFrame,
        n_rows: int,
        feature_name: str,
    ) -> tuple[str, np.ndarray]:
        """Transform one feature into its target-encoded vector."""
        te_col_name = f"TE_{self.aggregation.upper()}_{feature_name}"
        mapping = self.group_stats_[feature_name]
        encoded = np.empty(n_rows, dtype=np.float32)

        for start_idx in range(0, n_rows, self.chunk_size):
            end_idx = min(start_idx + self.chunk_size, n_rows)
            chunk = df.iloc[start_idx:end_idx]
            encoded[start_idx:end_idx] = self._encode_feature_chunk(
                chunk, feature_name, mapping
            )

        return te_col_name, encoded

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Transform the data using learned target encodings.

        Args:
            df (pd.DataFrame): Feature DataFrame to transform.

        Returns:
            pd.DataFrame: DataFrame with target-encoded features.

        Raises:
            ValueError: If the encoder has not been fitted yet.
        """
        if not self.group_stats_:
            raise ValueError(
                "TargetEncodingFeatureGenerator must be fitted before transform."
            )

        fitted_index = self._state["fitted_index"]
        fitted_te_df = self._state["fitted_te_df"]
        if (
            fitted_index is not None
            and fitted_te_df is not None
            and df.index.equals(fitted_index)
        ):
            return pd.concat([df, fitted_te_df], axis=1)

        n_rows = len(df)
        te_columns_dict: dict[str, np.ndarray] = {}
        pbar = tqdm(
            total=len(self.features_names), desc="Transforming with target encoding"
        )

        for feature_name in self.features_names:
            te_col_name, encoded = self._transform_single_feature(
                df, n_rows, feature_name
            )
            te_columns_dict[te_col_name] = encoded
            pbar.update(1)
            gc.collect()

        pbar.close()

        te_columns_df = pd.DataFrame(te_columns_dict, index=df.index)
        result_df = pd.concat([df, te_columns_df], axis=1)

        del te_columns_dict, te_columns_df
        gc.collect()

        return result_df

    def fit_transform(self, df: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
        """
        Fit the encoder and transform the data.

        Args:
            df (pd.DataFrame): Feature DataFrame.
            y (pd.Series): Target series.

        Returns:
            pd.DataFrame: DataFrame with target-encoded features.
        """
        return self.fit(df, y).transform(df)

    def get_feature_stats(
        self, df: pd.DataFrame, y: pd.Series, feature_name: str
    ) -> pd.DataFrame:
        """
        Compute statistics for the target variable grouped by a feature.

        Args:
            df (pd.DataFrame): Feature DataFrame.
            y (pd.Series): Target series.
            feature_name (str): Name of the feature to compute statistics for.

        Returns:
            pd.DataFrame: DataFrame with computed statistics.
        """
        return self._compute_feature_stats(df, y, feature_name)

    def _validate_fit_inputs(self, df: pd.DataFrame, y: pd.Series) -> None:
        """
        Validate inputs for fit and fit_transform.

        Args:
            df (pd.DataFrame): Feature DataFrame.
            y (pd.Series): Target series.

        Raises:
            TypeError: If df is not a DataFrame or y is not array-like.
            ValueError: If df and y have different lengths.
            KeyError: If features are missing from df.
        """
        if not isinstance(df, pd.DataFrame):
            raise TypeError("df must be a pandas DataFrame.")
        if not isinstance(y, (pd.Series, pd.DataFrame, np.ndarray, list, tuple)):
            raise TypeError("y must be a pandas Series or array-like.")
        y_len = len(y)
        if len(df) != y_len:
            raise ValueError(
                f"df and y must have the same number of rows. "
                f"Got len(df)={len(df)} and len(y)={y_len}."
            )
        missing = [
            feature for feature in self.features_names if feature not in df.columns
        ]
        if missing:
            raise KeyError(f"Missing features in df: {missing}")


if __name__ == "__main__":
    print("TargetEncodingFeatureGenerator module loaded.")
