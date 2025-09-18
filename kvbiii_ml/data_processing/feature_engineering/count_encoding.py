import pandas as pd
from tqdm import tqdm


class CountEncodingFeatureGenerator:
    """
    Class for generating count encoded features.
    """

    def __init__(self, features_names: list[str] | None = None, fill_value: int = 0):
        """
        Initialize the CountEncodingFeatureGenerator.

        Args:
            features_names (list[str]): List of feature names to be count encoded.
            fill_value (int): Value to fill for unseen categories during transform (default is 0).
        """
        self.features_names = features_names if features_names is not None else []
        self.fill_value = fill_value
        self.count_maps_ = {}
        self._validate_init_params()

    def _validate_init_params(self):
        """
        Validate the parameters for count encoding.
        """
        if not isinstance(self.features_names, list) or not all(
            isinstance(name, str) for name in self.features_names
        ):
            raise ValueError("features_names must be a list of strings.")
        if not isinstance(self.fill_value, (int, float)):
            raise ValueError("fill_value must be a numeric value.")

    def fit(self, df: pd.DataFrame) -> "CountEncodingFeatureGenerator":
        """
        Fit the CountEncodingFeatureGenerator by computing count maps for each feature.

        Args:
            df (pd.DataFrame): DataFrame to fit on.

        Returns:
            CountEncodingFeatureGenerator: Fitted instance.
        """
        self.count_maps_ = {}

        # If no features specified, use all columns from df
        if not self.features_names:
            self.features_names = list(df.columns)

        for feature_name in tqdm(
            self.features_names, desc="Computing count encoding maps"
        ):
            if feature_name not in df.columns:
                raise ValueError(
                    f"Feature '{feature_name}' not found among DataFrame features."
                )

            counts = df[feature_name].value_counts()
            self.count_maps_[feature_name] = counts.to_dict()

        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Transform the DataFrame with count encoding.

        Args:
            df (pd.DataFrame): DataFrame to be transformed.

        Returns:
            pd.DataFrame: DataFrame with the count-encoded features added.
        """
        if not self.count_maps_:
            raise ValueError(
                "CountEncodingFeatureGenerator must be fitted before transform."
            )

        result_df = df.copy()

        new_columns = {}
        for feature_name in tqdm(
            self.features_names, desc="Transforming with count encoding"
        ):
            count_map = self.count_maps_[feature_name]
            encoded_values = (
                df[feature_name].map(count_map).fillna(self.fill_value).astype("int32")
            )
            new_columns[f"CE_{feature_name}"] = encoded_values

        result_df = pd.concat(
            [result_df, pd.DataFrame(new_columns, index=df.index)], axis=1
        )

        return result_df

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Fit and transform the DataFrame with count encoding.

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

    def get_count_maps(self) -> dict:
        """
        Get the count maps for each feature.

        Returns:
            dict: Dictionary containing count maps for each feature.
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
