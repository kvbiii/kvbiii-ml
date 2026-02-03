import gc
from itertools import combinations

import numpy as np
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
        chunk_size: int = 50000,
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
            chunk_size (int, optional): Number of rows to process at once during
                transform. Defaults to 50000.
        """
        self.features_names = features_names if features_names is not None else []
        self.degree = degree
        self.separator = separator
        self.batch_size = batch_size
        self.chunk_size = chunk_size
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

    def _create_combined_string(
        self, df: pd.DataFrame, columns: list[str]
    ) -> np.ndarray:
        """
        Create combined string efficiently using numpy operations.

        Args:
            df (pd.DataFrame): DataFrame containing the columns.
            columns (list[str]): List of column names to combine.

        Returns:
            np.ndarray: Array of combined strings.
        """
        arrays = [df[col].astype(str).values for col in columns]

        if len(arrays) == 2:
            return np.char.add(np.char.add(arrays[0], self.separator), arrays[1])

        result = arrays[0].copy()
        for arr in arrays[1:]:
            result = np.char.add(np.char.add(result, self.separator), arr)
        return result

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
            combo_name = "X".join(combo)
            combo_list = list(combo)
            is_numerical = self._is_numerical_combo(df, combo_list)

            if is_numerical:
                self.numerical_combos_.add(combo_name)
            else:
                combined = self._create_combined_string(df, combo_list)
                unique_vals = np.unique(combined)
                self.encoding_maps_[combo_name] = {
                    val: idx for idx, val in enumerate(unique_vals)
                }
                del combined, unique_vals

            if len(self.encoding_maps_) % 10 == 0:
                gc.collect()

        gc.collect()
        return self

    def _transform_chunk(
        self, chunk: pd.DataFrame, combo: tuple[str, ...]
    ) -> pd.Series:
        """
        Transform a single chunk for a given combination.

        Args:
            chunk (pd.DataFrame): DataFrame chunk to transform.
            combo (tuple[str, ...]): Feature combination tuple.

        Returns:
            pd.Series: Transformed values for the chunk.
        """
        combo_name = "X".join(combo)
        combo_list = list(combo)

        if combo_name in self.numerical_combos_:
            result = chunk[combo_list].prod(axis=1).astype("float32")
        else:
            encoding_map = self.encoding_maps_[combo_name]
            combined = self._create_combined_string(chunk, combo_list)
            result = pd.Series(
                [encoding_map.get(val, -1) for val in combined],
                index=chunk.index,
                dtype="int32",
            )
            del combined

        return result

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

        n_rows = len(df)
        n_chunks = (n_rows + self.chunk_size - 1) // self.chunk_size

        new_columns_dict = {
            "X".join(combo): np.empty(
                n_rows,
                dtype=(
                    np.float32
                    if "X".join(combo) in self.numerical_combos_
                    else np.int32
                ),
            )
            for combo in self.feature_combinations_
        }

        total_combos = len(self.feature_combinations_)
        pbar_combos = tqdm(
            total=total_combos, desc=f"Transforming {self.degree}-way combinations"
        )

        for combo_idx in range(0, total_combos, self.batch_size):
            batch_combos = self.feature_combinations_[
                combo_idx : combo_idx + self.batch_size
            ]

            for chunk_idx in range(n_chunks):
                start_idx = chunk_idx * self.chunk_size
                end_idx = min((chunk_idx + 1) * self.chunk_size, n_rows)
                chunk = df.iloc[start_idx:end_idx]

                for combo in batch_combos:
                    combo_name = "X".join(combo)
                    result = self._transform_chunk(chunk, combo)
                    new_columns_dict[combo_name][start_idx:end_idx] = result.values
                    del result

                del chunk
                gc.collect()

            pbar_combos.update(len(batch_combos))

        pbar_combos.close()

        new_df = pd.DataFrame(new_columns_dict, index=df.index)
        result_df = pd.concat([df, new_df], axis=1)

        del new_columns_dict, new_df
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
            ["X".join(combo) for combo in self.feature_combinations_]
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
