"""Module for count encoding feature generation."""

import gc

import pandas as pd
from tqdm import tqdm


class CountEncodingFeatureGenerator:
    """Class for generating count encoded features."""

    def __init__(
        self,
        features_names: list[str] | None = None,
        fill_value: int = 0,
        batch_size: int = 20,
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
        """
        self.features_names = features_names if features_names is not None else []
        self.fill_value = fill_value
        self.batch_size = batch_size
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

        self.count_maps_ = {
            feature_name: df[feature_name].value_counts().to_dict()
            for feature_name in tqdm(
                features_to_process,
                desc="Computing count encoding maps",
            )
        }

        gc.collect()
        return self

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

        new_columns = {}
        total_features = len(self.features_names)

        for i in tqdm(
            range(0, total_features, self.batch_size),
            desc="Transforming with count encoding",
        ):
            batch_features = self.features_names[i : i + self.batch_size]

            for feature_name in batch_features:
                count_map = self.count_maps_[feature_name]
                new_columns[f"CE_{feature_name}"] = (
                    df[feature_name]
                    .map(count_map)
                    .fillna(self.fill_value)
                    .astype("int32")
                )

            if i % (self.batch_size * 3) == 0:
                gc.collect()

        result_df = pd.concat([df, pd.DataFrame(new_columns, index=df.index)], axis=1)
        del new_columns
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


# Backward compatible alias expected by some tests
CountEncoder = CountEncodingFeatureGenerator  # pragma: no cover - alias


if __name__ == "__main__":
    # Example usage
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
