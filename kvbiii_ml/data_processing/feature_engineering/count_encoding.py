import pandas as pd
from tqdm import tqdm


class CountEncodingFeatureGenerator:
    """
    Class for generating count encoded features.
    """

    def __init__(self, features_names: list[str] | None = None, fill_value: int = 0):
        """Initialize the CountEncodingFeatureGenerator.

        Args:
            features_names (list[str] | None, optional): List of feature names to be count encoded.
                If None, all columns will be used. Defaults to None.
            fill_value (int, optional): Value for unseen categories during transform. Defaults to 0.
        """
        self.features_names = features_names if features_names is not None else []
        self.fill_value = fill_value
        self.count_maps_: dict[str, dict] = {}
        self._validate_init_params()

    def _validate_init_params(self):
        """Validate the initialization parameters."""
        if not isinstance(self.features_names, list) or not all(
            isinstance(name, str) for name in self.features_names
        ):
            raise ValueError("features_names must be a list of strings.")
        if not isinstance(self.fill_value, (int, float)):
            raise ValueError("fill_value must be a numeric value.")

    def fit(self, df: pd.DataFrame) -> "CountEncodingFeatureGenerator":
        """Fit the encoder by computing count maps for each feature.

        Args:
            df (pd.DataFrame): DataFrame to fit on.

        Returns:
            CountEncodingFeatureGenerator: The fitted instance.
        """
        self.count_maps_ = {}
        features_to_process = self.features_names or list(df.columns)
        self.features_names = features_to_process

        for feature_name in tqdm(
            features_to_process, desc="Computing count encoding maps"
        ):
            if feature_name not in df.columns:
                raise ValueError(
                    f"Feature '{feature_name}' not found in DataFrame columns."
                )
            self.count_maps_[feature_name] = df[feature_name].value_counts().to_dict()

        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Transform the DataFrame with count encoding.

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

        result_df = df.copy()
        for feature_name in tqdm(
            self.features_names, desc="Transforming with count encoding"
        ):
            count_map = self.count_maps_[feature_name]
            result_df[f"CE_{feature_name}"] = (
                df[feature_name].map(count_map).fillna(self.fill_value).astype("int32")
            )

        return result_df

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Fit and transform the DataFrame with count encoding in one step.

        Args:
            df (pd.DataFrame): DataFrame to fit and transform.

        Returns:
            pd.DataFrame: DataFrame with the count-encoded features added.
        """
        self.count_maps_ = {}
        result_df = df.copy()
        features_to_process = self.features_names or list(df.columns)
        self.features_names = features_to_process

        for feature_name in tqdm(
            features_to_process, desc="Fitting and transforming count encoding"
        ):
            if feature_name not in df.columns:
                raise ValueError(
                    f"Feature '{feature_name}' not found in DataFrame columns."
                )
            count_map = df[feature_name].value_counts().to_dict()
            self.count_maps_[feature_name] = count_map
            result_df[f"CE_{feature_name}"] = (
                df[feature_name].map(count_map).fillna(self.fill_value).astype("int32")
            )

        return result_df

    def get_feature_names(self) -> list[str]:
        """Get the names of the generated count encoded features.

        Returns:
            list[str]: List of generated feature names.
        """
        return [f"CE_{feature_name}" for feature_name in self.features_names]

    def get_count_maps(self) -> dict[str, dict]:
        """Get the count maps for each feature.

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
    df = pd.DataFrame(data)

    count_encoder = CountEncodingFeatureGenerator(features_names=["A", "B", "C"])
    transformed_df = count_encoder.fit_transform(df)
    print(transformed_df)
    test_data = {"A": ["x", "b", "c"], "B": ["x", "y", "z"], "C": ["1", "s", "3"]}
    test_df = pd.DataFrame(test_data)
    transformed_test_df = count_encoder.transform(test_df)
    print(transformed_test_df)
