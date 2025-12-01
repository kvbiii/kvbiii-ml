"""Module for cross-feature encoding through feature combinations."""

import gc
from itertools import combinations

import pandas as pd
from tqdm import tqdm


class CrossFeatureGenerator:
    """Class for generating cross-encoded features from feature combinations."""

    def __init__(
        self,
        features_names: list[str] | None = None,
        degree: int = 2,
        separator: str = "_",
        batch_size: int = 10,
    ):
        """
        Initialize the CrossFeatureGenerator.

        Args:
            features_names (list[str] | None, optional): List of feature names to
                create combinations from. If None, all columns will be used.
                Defaults to None.
            degree (int, optional): Degree of combinations to generate. Defaults to 2.
            separator (str, optional): Separator for combining feature values.
                Defaults to '_':
            batch_size (int, optional): Number of combinations to process at once.
                Smaller values use less memory. Defaults to 10.
        """
        self.features_names = features_names if features_names is not None else []
        self.degree = degree
        self.separator = separator
        self.batch_size = batch_size
        self.feature_combinations_ = []
        self.encoding_maps_ = {}
        self.numerical_combos_ = set()
        self._validate_init_params()

    def _validate_init_params(self):
        """
        Validate the parameters for cross feature generation.

        Raises:
            ValueError: If parameters are invalid.
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
                f"Number of features ({len(self.features_names)}) must be >= "
                f"degree ({self.degree})."
            )

    def _is_numerical_combo(self, df: pd.DataFrame, columns: list[str]) -> bool:
        """
        Check if all columns in the combination are numerical.

        Args:
            df (pd.DataFrame): DataFrame to check.
            columns (list[str]): List of column names to check.

        Returns:
            bool: True if all columns are numerical, False otherwise.
        """
        return all(pd.api.types.is_numeric_dtype(df[col]) for col in columns)

    def fit(self, df: pd.DataFrame) -> "CrossFeatureGenerator":
        """
        Fit the CrossFeatureGenerator by computing encoding maps.

        Args:
            df (pd.DataFrame): DataFrame to fit on.

        Returns:
            CrossFeatureGenerator: Fitted instance.

        Raises:
            ValueError: If number of features is less than degree.
        """
        if not self.features_names:
            self.features_names = list(df.columns)
            if len(self.features_names) < self.degree:
                raise ValueError(
                    f"Number of features ({len(self.features_names)}) must be >= "
                    f"degree ({self.degree})."
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
            combo_list = list(combo)
            is_numerical = self._is_numerical_combo(df, combo_list)

            if is_numerical:
                self.numerical_combos_.add(combo_name)
            else:
                unique_combos = df[combo_list].drop_duplicates()
                encoding_map = {
                    self.separator.join([str(value) for value in row]): idx
                    for idx, row in enumerate(unique_combos.itertuples(index=False))
                }
                self.encoding_maps_[combo_name] = encoding_map
                del unique_combos

        gc.collect()
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Transform the DataFrame by adding cross-encoded features.

        Args:
            df (pd.DataFrame): DataFrame to transform.

        Returns:
            pd.DataFrame: DataFrame with cross-encoded features added.

        Raises:
            ValueError: If the generator has not been fitted.
        """
        if (
            not hasattr(self, "numerical_combos_")
            or (not self.encoding_maps_ and not self.numerical_combos_)
            or not self.feature_combinations_
        ):
            raise ValueError("CrossFeatureGenerator must be fitted before transform.")

        new_columns = {}
        total_combos = len(self.feature_combinations_)

        for i in tqdm(
            range(0, total_combos, self.batch_size),
            desc=f"Transforming {self.degree}-way combinations",
        ):
            batch_combos = self.feature_combinations_[i : i + self.batch_size]

            for combo in batch_combos:
                combo_name = "-".join(combo)
                combo_list = list(combo)

                if combo_name in self.numerical_combos_:
                    new_columns[combo_name] = (
                        df[combo_list].prod(axis=1).astype("float32")
                    )
                else:
                    encoding_map = self.encoding_maps_[combo_name]
                    df_subset = df[combo_list]
                    combined = df_subset.astype(str).agg(
                        self.separator.join, axis=1
                    )
                    new_columns[combo_name] = (
                        combined.map(encoding_map).fillna(-1).astype("int32")
                    )
                    del df_subset, combined

            if i % (self.batch_size * 5) == 0:
                gc.collect()

        result_df = pd.concat(
            [df, pd.DataFrame(new_columns, index=df.index)], axis=1
        )
        del new_columns
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
        return self.fit(df).transform(df)

    def get_feature_names(self) -> list[str]:
        """
        Get the names of the generated cross features.

        Returns:
            list[str]: List of generated feature names.
        """
        return (
            ["-".join(combo) for combo in self.feature_combinations_]
            if self.feature_combinations_
            else []
        )


if __name__ == "__main__":
    data = {
        "A": ["a", "b", "a", "c"],
        "B": ["x", "y", "x", "z"],
        "C": ["1", "2", "1", "3"],
        "D": [10, 20, 10, 30],
        "E": [0.1, 0.2, 0.1, 0.3],
    }
    df_example = pd.DataFrame(data)

    generator = CrossFeatureGenerator(
        features_names=["A", "B", "C", "D", "E"], degree=2, batch_size=5
    )
    transformed_df_example = generator.fit_transform(df_example)
    print(transformed_df_example)

    test_data = {
        "A": ["x", "b", "c"],
        "B": ["x", "y", "z"],
        "C": ["1", "s", "3"],
        "D": [15, 20, 30],
        "E": [0.5, 0.2, 0.3],
    }
    test_df_example = pd.DataFrame(test_data)
    transformed_test_df_example = generator.transform(test_df_example)
    print(transformed_test_df_example)
    print("Generated feature names:", generator.get_feature_names())
