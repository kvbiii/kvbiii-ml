import pandas as pd
from itertools import combinations
from tqdm import tqdm
import gc


class CrossFeatureGenerator:
    """
    Class for generating cross-encoded features by creating combinations of categorical features.
    """

    def __init__(
        self, features_names: list[str] = [], degree: int = 2, separator: str = "_"
    ):
        """
        Initialize the CrossFeatureGenerator.

        Args:
            features_names (list[str]): List of feature names to create combinations from. If empty, all columns from the DataFrame will be used.
            degree (int): Degree of combinations to generate (default is 2).
            separator (str): Separator to use when combining feature values (default is '_').
        """
        self.features_names = features_names
        self.degree = degree
        self.separator = separator
        self.feature_combinations_ = []
        self.encoding_maps_ = {}
        self.numerical_combos_ = set()  # Track which combinations are numerical
        self._validate_init_params()

    def _validate_init_params(self):
        """
        Validate the parameters for cross feature generation.
        """
        if not isinstance(self.features_names, list) or not all(
            isinstance(name, str) for name in self.features_names
        ):
            raise ValueError("features_names must be a list of strings.")
        if not isinstance(self.degree, int) or self.degree < 2:
            raise ValueError("degree must be an integer >= 2.")
        if not isinstance(self.separator, str):
            raise ValueError("separator must be a string.")
        if self.features_names and len(self.features_names) < self.degree:
            raise ValueError(
                f"Number of features ({len(self.features_names)}) must be >= degree ({self.degree})."
            )

    def _is_numerical_combo(self, df: pd.DataFrame, columns: list) -> bool:
        """
        Check if all columns in the combination are numerical.

        Args:
            df (pd.DataFrame): DataFrame to check
            columns (list): List of column names to check

        Returns:
            bool: True if all columns are numerical, False otherwise
        """
        return all(pd.api.types.is_numeric_dtype(df[col]) for col in columns)

    def fit(self, df: pd.DataFrame) -> "CrossFeatureGenerator":
        """
        Fit the CrossFeatureGenerator by computing encoding maps for feature combinations.

        Args:
            df (pd.DataFrame): DataFrame to fit on.

        Returns:
            CrossFeatureGenerator: Fitted instance.
        """
        if not self.features_names:
            self.features_names = list(df.columns)
            if len(self.features_names) < self.degree:
                raise ValueError(
                    f"Number of features ({len(self.features_names)}) must be >= degree ({self.degree})."
                )

        self.feature_combinations_ = list(
            combinations(sorted(self.features_names), self.degree)
        )
        self.encoding_maps_ = {}
        self.numerical_combos_ = set()

        for combo in tqdm(
            self.feature_combinations_, desc=f"Fitting {self.degree}-way combinations"
        ):
            combo_name = "-".join(combo)
            is_numerical = self._is_numerical_combo(df, list(combo))

            if is_numerical:
                self.numerical_combos_.add(combo_name)
            else:
                unique_combinations = (
                    df[list(combo)].drop_duplicates().itertuples(index=False)
                )
                encoding_map = {
                    self.separator.join(map(str, row)): i
                    for i, row in enumerate(unique_combinations)
                }
                self.encoding_maps_[combo_name] = encoding_map
        gc.collect()
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Transform the DataFrame by adding cross-encoded features.

        Args:
            df (pd.DataFrame): DataFrame to transform.

        Returns:
            pd.DataFrame: DataFrame with cross-encoded features added.
        """
        if (
            not hasattr(self, "numerical_combos_")
            or (not self.encoding_maps_ and not self.numerical_combos_)
            or not self.feature_combinations_
        ):
            raise ValueError("CrossFeatureGenerator must be fitted before transform.")

        result_df = df.copy()

        for combo in tqdm(
            self.feature_combinations_,
            desc=f"Transforming {self.degree}-way combinations",
        ):
            combo_name = "-".join(combo)
            combo_list = list(combo)

            if combo_name in self.numerical_combos_:
                result_df[combo_name] = df[combo_list].prod(axis=1)
            else:
                encoding_map = self.encoding_maps_[combo_name]
                combined_values = (
                    df[combo_list].astype(str).agg(self.separator.join, axis=1)
                )
                result_df[combo_name] = (
                    combined_values.map(encoding_map).fillna(-1).astype("int32")
                )
                del combined_values
                gc.collect()
        gc.collect()
        return result_df

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Fit and transform the DataFrame with cross-encoded features.

        Args:
            df (pd.DataFrame): DataFrame to fit and transform.

        Returns:
            pd.DataFrame: DataFrame with cross-encoded features added.
        """
        self.fit(df)
        return self.transform(df)

    def get_feature_names(self) -> list[str]:
        """
        Get the names of the generated cross features.

        Returns:
            list[str]: List of generated feature names.
        """
        if not self.feature_combinations_:
            return []
        return ["-".join(combo) for combo in self.feature_combinations_]


if __name__ == "__main__":
    # Example usage
    data = {
        "A": ["a", "b", "a", "c"],
        "B": ["x", "y", "x", "z"],
        "C": ["1", "2", "1", "3"],
        "D": [10, 20, 10, 30],
        "E": [0.1, 0.2, 0.1, 0.3],
    }
    df = pd.DataFrame(data)

    generator = CrossFeatureGenerator(
        features_names=["A", "B", "C", "D", "E"], degree=2
    )
    transformed_df = generator.fit_transform(df)
    print(transformed_df)
    test_data = {
        "A": ["x", "b", "c"],
        "B": ["x", "y", "z"],
        "C": ["1", "s", "3"],
        "D": [15, 20, 30],
        "E": [0.5, 0.2, 0.3],
    }
    test_df = pd.DataFrame(test_data)
    transformed_test_df = generator.transform(test_df)
    print(transformed_test_df)
    print("Generated feature names:", generator.get_feature_names())
