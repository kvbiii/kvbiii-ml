import gc

import numpy as np
import pandas as pd
from tqdm import tqdm


class CountEncodingFeatureGenerator:
    """Class for generating count encoded features."""

    def __init__(
        self,
        features_names: list[str] | None = None,
        fill_value: int = 0,
        batch_size: int = 20,
        chunk_size: int = 50000,
    ):
        """
        Initialize the CountEncodingFeatureGenerator.

        Args:
            features_names (list[str] | None, optional): List of feature names to
                be count encoded. If None, all columns will be used. Defaults to None.
            fill_value (int, optional): Value for unseen categories during transform.
                Defaults to 0.
            batch_size (int, optional): Number of features to process at once.
                Defaults to 20.
            chunk_size (int, optional): Number of rows to process at once during
                transform. Defaults to 50000.
        """
        self.features_names = features_names if features_names is not None else []
        self.fill_value = fill_value
        self.batch_size = batch_size
        self.chunk_size = chunk_size
        self.count_maps_: dict[str, dict] = {}
        self._validate_init_params()

    def _validate_init_params(self):
        """
        Validate the initialization parameters.

        Raises:
            ValueError: If parameters are invalid.
        """
        if not isinstance(self.features_names, list) or not all(
            isinstance(name, str) for name in self.features_names
        ):
            raise ValueError("features_names must be a list of strings.")
        if not isinstance(self.fill_value, (int, float)):
            raise ValueError("fill_value must be a numeric value.")
        if not isinstance(self.batch_size, int) or self.batch_size < 1:
            raise ValueError("batch_size must be a positive integer.")
        if not isinstance(self.chunk_size, int) or self.chunk_size < 1:
            raise ValueError("chunk_size must be a positive integer.")

    def fit(self, df: pd.DataFrame) -> "CountEncodingFeatureGenerator":
        """
        Fit the encoder by computing count maps for each feature.

        Args:
            df (pd.DataFrame): DataFrame to fit on.

        Returns:
            CountEncodingFeatureGenerator: The fitted instance.

        Raises:
            ValueError: If a feature name is not found in DataFrame columns.
        """
        features_to_process = self.features_names or list(df.columns)
        self.features_names = features_to_process

        for feature_name in features_to_process:
            if feature_name not in df.columns:
                raise ValueError(
                    f"Feature '{feature_name}' not found in DataFrame columns."
                )

        self.count_maps_ = {}
        for feature_name in tqdm(
            features_to_process,
            desc="Computing count encoding maps",
        ):
            counts = df[feature_name].value_counts()
            self.count_maps_[feature_name] = counts.to_dict()
            del counts

            if len(self.count_maps_) % 10 == 0:
                gc.collect()

        gc.collect()
        return self

    def _transform_feature_chunk(
        self,
        chunk: pd.DataFrame,
        feature_name: str,
    ) -> np.ndarray:
        """
        Transform a single chunk for a given feature.

        Args:
            chunk (pd.DataFrame): DataFrame chunk to transform.
            feature_name (str): Name of the feature to encode.

        Returns:
            np.ndarray: Array of count-encoded values.
        """
        count_map = self.count_maps_[feature_name]
        values = chunk[feature_name].values

        result = np.array(
            [count_map.get(val, self.fill_value) for val in values],
            dtype=np.int32,
        )
        return result

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Transform the DataFrame with count encoding.

        Args:
            df (pd.DataFrame): DataFrame to be transformed.

        Returns:
            pd.DataFrame: DataFrame with the count-encoded features added.

        Raises:
            ValueError: If the encoder has not been fitted yet.
        """
        if not self.count_maps_:
            raise ValueError(
                "CountEncodingFeatureGenerator must be fitted before transform."
            )

        n_rows = len(df)
        n_chunks = (n_rows + self.chunk_size - 1) // self.chunk_size

        new_columns_dict = {
            f"CE_{feature_name}": np.empty(n_rows, dtype=np.int32)
            for feature_name in self.features_names
        }

        total_features = len(self.features_names)
        pbar = tqdm(
            total=total_features,
            desc="Transforming with count encoding",
        )

        for feat_idx in range(0, total_features, self.batch_size):
            batch_features = self.features_names[feat_idx : feat_idx + self.batch_size]

            for chunk_idx in range(n_chunks):
                start_idx = chunk_idx * self.chunk_size
                end_idx = min((chunk_idx + 1) * self.chunk_size, n_rows)
                chunk = df.iloc[start_idx:end_idx]

                for feature_name in batch_features:
                    result = self._transform_feature_chunk(chunk, feature_name)
                    new_columns_dict[f"CE_{feature_name}"][start_idx:end_idx] = result
                    del result

                del chunk
                gc.collect()

            pbar.update(len(batch_features))

        pbar.close()

        new_df = pd.DataFrame(new_columns_dict, index=df.index)
        result_df = pd.concat([df, new_df], axis=1)

        del new_columns_dict, new_df
        gc.collect()

        return result_df

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Fit and transform the DataFrame with count encoding in one step.

        Args:
            df (pd.DataFrame): DataFrame to fit and transform.

        Returns:
            pd.DataFrame: DataFrame with the count-encoded features added.
        """
        return self.fit(df).transform(df)

    def get_feature_names(self) -> list[str]:
        """
        Get the names of the generated count encoded features.

        Returns:
            list[str]: List of generated feature names.
        """
        return [f"CE_{feature_name}" for feature_name in self.features_names]

    def get_count_maps(self) -> dict[str, dict]:
        """
        Get the count maps for each feature.

        Returns:
            dict[str, dict]: A dictionary of count maps for each feature.
        """
        return self.count_maps_.copy()


CountEncoder = CountEncodingFeatureGenerator


if __name__ == "__main__":
    data = {
        "A": ["a", "b", "a", "c"],
        "B": ["x", "y", "x", "z"],
        "C": ["1", "2", "1", "3"],
    }
    df_temp = pd.DataFrame(data)

    count_encoder = CountEncodingFeatureGenerator(
        features_names=["A", "B", "C"], batch_size=2
    )
    transformed_df = count_encoder.fit_transform(df_temp)
    print(transformed_df)
    test_data = {"A": ["x", "b", "c"], "B": ["x", "y", "z"], "C": ["1", "s", "3"]}
    test_df = pd.DataFrame(test_data)
    transformed_test_df = count_encoder.transform(test_df)
    print(transformed_test_df)
