"""
Data cleaning utilities for preprocessing DataFrames.

This module provides comprehensive data cleaning functionality including
feature removal, missing value analysis, and categorical feature detection.
Optimized for large datasets with efficient vectorized operations.
"""

from collections import defaultdict
import numpy as np
import pandas as pd


class DataCleaner:
    """
    A comprehensive data cleaning class for preprocessing DataFrames.

    This class provides methods for cleaning and preprocessing data including
    removing unnecessary features, handling missing values, optimizing data types,
    and identifying categorical features. All methods are optimized for large datasets.
    """

    @staticmethod
    def get_categorical_features(df: pd.DataFrame, threshold: int = 100) -> list[str]:
        """
        Gets categorical features from the DataFrame.

        Identifies features as categorical based on data type and number of unique values.
        Binary features (2 unique values) are automatically included regardless of type.

        Args:
            df (pd.DataFrame): The input DataFrame.
            threshold (int, optional): Maximum number of unique values for a feature
                to be considered categorical. Defaults to 100.

        Returns:
            list[str]: List of categorical feature names.
        """
        categorical_cols = df.select_dtypes(include=["object", "category"]).columns
        nunique = df.nunique()

        categorical_features = [
            col for col in categorical_cols if nunique[col] < threshold
        ]
        binary_features = [col for col in df.columns if nunique[col] == 2]

        return list(set(categorical_features) | set(binary_features))

    @staticmethod
    def initial_features_removal(df: pd.DataFrame) -> pd.DataFrame:
        """
        Removes features that are not useful for modeling.

        Removes features with:
        - Only 1 unique value (variance = 0)
        - All missing values
        - Duplicated features

        Args:
            df (pd.DataFrame): DataFrame to be cleaned.

        Returns:
            pd.DataFrame: Cleaned DataFrame with unnecessary features removed.
        """
        initial_feature_count = df.shape[1]
        removed_features: list[str] = []

        # Phase 1: Remove features with only 1 unique value (O(n) operations)
        unique_counts = df.nunique(dropna=False)
        single_unique_mask = unique_counts == 1
        single_unique_features = df.columns[single_unique_mask].tolist()
        if single_unique_features:
            removed_features.extend(single_unique_features)
            df = df.loc[:, ~single_unique_mask]

        # Phase 2: Remove features with all missing values
        if not df.empty:
            all_missing_mask = df.isna().all()
            all_missing_features = df.columns[all_missing_mask].tolist()
            if all_missing_features:
                removed_features.extend(all_missing_features)
                df = df.loc[:, ~all_missing_mask]

        # Phase 3: Remove duplicated features (optimized using hashing)
        if not df.empty:
            df, duplicate_features = DataCleaner._remove_duplicate_features(df)
            removed_features.extend(duplicate_features)

        # Phase 4: Remove bijective features
        if not df.empty:
            df, bijective_features = DataCleaner.remove_bijective_features(df)
            removed_features.extend(bijective_features)
        final_feature_count = df.shape[1]
        features_removed = initial_feature_count - final_feature_count

        if removed_features:
            print(f"Removed {features_removed} features: {removed_features}")

        print(
            f"Features after removal: {final_feature_count}. "
            f"Removed {features_removed} features total."
        )

        return df

    @staticmethod
    def _remove_duplicate_features(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
        """
        Efficiently identifies and removes duplicate features.

        Uses hashing for O(n) complexity instead of O(n²) pairwise comparisons.

        Args:
            df (pd.DataFrame): DataFrame to deduplicate.

        Returns:
            tuple[pd.DataFrame, list[str]]: Tuple of (cleaned DataFrame, list of removed features).
        """
        features = list(df.columns)
        feature_hash_map: dict[int, str] = {}
        columns_to_keep: list[str] = []
        removed_features: list[str] = []

        for feature in features:
            try:
                feature_hash = pd.util.hash_array(df[feature].values).tobytes()
                int_hash = hash(feature_hash)
            except (TypeError, AttributeError):
                int_hash = hash(str(df[feature].values.tobytes()))
            if int_hash in feature_hash_map:
                original_feature = feature_hash_map[int_hash]
                if df[feature].equals(df[original_feature]):
                    removed_features.append(feature)
                    print(
                        f"Feature '{feature}' duplicates '{original_feature}'. Removing '{feature}'."
                    )
                    continue
            feature_hash_map[int_hash] = feature
            columns_to_keep.append(feature)

        return df[columns_to_keep], removed_features

    @staticmethod
    def drop_highly_skewed_categorical_features(
        df: pd.DataFrame, features_to_investigate: list[str], threshold: float = 0.99
    ) -> pd.DataFrame:
        """
        Drops categorical features with highly skewed distributions.

        Removes features where one value accounts for more than the specified
        threshold of the data.

        Args:
            df (pd.DataFrame): The input DataFrame.
            features_to_investigate (list[str]): List of categorical features
                to investigate.
            threshold (float, optional): The threshold for skewness in the
                distribution. Defaults to 0.99.

        Returns:
            pd.DataFrame: The DataFrame with highly skewed categorical features dropped.
        """
        feats_to_drop = []
        for feature in features_to_investigate:
            value_counts = df[feature].value_counts(normalize=True, dropna=False)
            max_value_ratio = value_counts.max()
            if max_value_ratio > threshold:
                max_value = value_counts.idxmax()
                print(
                    f"  - {feature} (value '{max_value}' with "
                    f"{max_value_ratio*100:.2f}% of the data)"
                )
                feats_to_drop.append(feature)

        if not feats_to_drop:
            return df

        print(
            f"Highly skewed categorical features with a distribution where "
            f"one value is more than {threshold*100:.2f}% of the data:\n"
        )
        df = df.drop(columns=feats_to_drop)
        print(
            f"\nNumber of features after removal: {df.shape[1]}. "
            f"Removed {len(feats_to_drop)} features."
        )
        return df

    @staticmethod
    def remove_bijective_features(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
        """
        Removes bijective features from the DataFrame.

        Detects and removes columns that are a bijective (one-to-one) mapping of another
        column, even if the values differ (e.g., label-encoded or mapped columns).

        Args:
            df (pd.DataFrame): The input DataFrame.

        Returns:
            tuple[pd.DataFrame, list[str]]: Tuple of (cleaned DataFrame, list of removed features).
        """
        columns = df.columns.tolist()
        columns_to_remove = set()
        unique_counts = {}
        candidates = []

        for col in columns:
            n_unique = df[col].nunique(dropna=False)
            if 1 < n_unique <= 1000:
                unique_counts[col] = n_unique
                candidates.append(col)
        count_groups = defaultdict(list)
        for col in candidates:
            count_groups[unique_counts[col]].append(col)
        for n_unique, cols_in_group in count_groups.items():
            if len(cols_in_group) < 2:
                continue
            factorized = {}
            valid_masks = {}
            for col in cols_in_group:
                if col in columns_to_remove:
                    continue
                mask = df[col].notna()
                if mask.any():
                    codes, _ = pd.factorize(df[col][mask])
                    factorized[col] = codes
                    valid_masks[col] = mask
            group_list = list(factorized.keys())
            for i, col1 in enumerate(group_list):
                if col1 in columns_to_remove:
                    continue
                codes1 = factorized[col1]
                mask1 = valid_masks[col1]
                for j in range(i + 1, len(group_list)):
                    col2 = group_list[j]
                    if col2 in columns_to_remove:
                        continue
                    codes2 = factorized[col2]
                    mask2 = valid_masks[col2]
                    combined_mask = mask1 & mask2
                    if not combined_mask.any():
                        continue
                    if not mask1.equals(mask2):
                        codes1_aligned = pd.factorize(df[col1][combined_mask])[0]
                        codes2_aligned = pd.factorize(df[col2][combined_mask])[0]
                    else:
                        codes1_aligned = codes1
                        codes2_aligned = codes2
                    if len(codes1_aligned) != len(codes2_aligned):
                        continue
                    mapping = {}
                    reverse_mapping = {}
                    is_bijective = True

                    for c1, c2 in zip(codes1_aligned, codes2_aligned):
                        if c1 in mapping:
                            if mapping[c1] != c2:
                                is_bijective = False
                                break
                        else:
                            if c2 in reverse_mapping:
                                is_bijective = False
                                break
                            mapping[c1] = c2
                            reverse_mapping[c2] = c1

                    if is_bijective and len(mapping) == n_unique:
                        print(
                            f"Features '{col1}' and '{col2}' are bijectively mapped. Removing '{col2}'."
                        )
                        columns_to_remove.add(col2)

        removed_features = list(columns_to_remove)
        if removed_features:
            df = df.drop(columns=removed_features)
            print(f"Removed {len(removed_features)} bijective features.")

        return df, removed_features
    
    @staticmethod
    def categorize_categorical_features_by_missing(
        data: pd.DataFrame, categorical_features: list[str]
    ) -> dict[str, list[str]]:
        """
        Categorizes categorical features by their proportion of missing values.

        Args:
            data (pd.DataFrame): The input DataFrame.
            categorical_features (list[str]): List of categorical features.

        Returns:
            dict[str, list[str]]: Dictionary with keys as types of categorical
                features and values as lists of feature names.
        """
        missing_ratio = data[categorical_features].isnull().mean()

        return {
            "categorical_not_many_missing_features": sorted(
                [col for col in categorical_features if 0 < missing_ratio[col] <= 0.1],
                key=lambda col: missing_ratio[col],
            ),
            "categorical_many_missing_features": sorted(
                [col for col in categorical_features if missing_ratio[col] > 0.1],
                key=lambda col: missing_ratio[col],
            ),
            "categorical_missing_features": sorted(
                [col for col in categorical_features if missing_ratio[col] > 0.0],
                key=lambda col: missing_ratio[col],
            ),
            "non_missing_categorical_features": sorted(
                [col for col in categorical_features if missing_ratio[col] == 0.0]
            ),
        }


def _demonstrate_usage() -> None:
    """Demonstrates the usage of DataCleaner class with sample data."""
    N: int = 10000
    sample_data = pd.DataFrame(
        {
            "numerical_feature": np.random.randint(0, 100, N),
            "categorical_feature": np.random.choice(["A", "B", "C", "D"], N),
            "duplicate_feature": np.random.randint(0, 100, N),
            "single_value_feature": ["constant"] * N,
            "all_na_feature": [np.nan] * N,
            "high_cardinality_feature": np.random.choice(range(500), N),
            "skewed_cat_feature": np.random.choice(
                ["majority", "minority"], N, p=[0.995, 0.005]
            ),
        }
    )

    sample_data["duplicate_feature"] = sample_data["numerical_feature"].copy()
    sample_data.loc[50:60, "numerical_feature"] = np.nan
    sample_data["bijective_feature"] = sample_data["categorical_feature"].map(
        {"A": "W", "B": "X", "C": "Y", "D": "Z"}
    )
    N_FEATURES = 2000
    new_features = {
        f"feature_{i}": np.random.randint(0, 10, N) for i in range(N_FEATURES)
    }
    sample_data = pd.concat([sample_data, pd.DataFrame(new_features)], axis=1)

    print("Original DataFrame shape:", sample_data.shape)
    print(
        "Original memory usage:",
        f"{sample_data.memory_usage(deep=True).sum() / 1024**2:.2f} MB",
    )

    cleaner = DataCleaner()

    cleaned_data = cleaner.initial_features_removal(sample_data)

    cat_feats = cleaner.get_categorical_features(cleaned_data, threshold=50)
    print(f"\nCategorical features (threshold=50): {cat_feats}")

    cleaned_data = cleaner.drop_highly_skewed_categorical_features(
        cleaned_data, cat_feats, threshold=0.99
    )

    cat_feats = cleaner.get_categorical_features(cleaned_data, threshold=50)

    cat_categories = cleaner.categorize_categorical_features_by_missing(
        cleaned_data, cat_feats
    )
    print("\nCategorical features categorized by missing values:")
    for key, value in cat_categories.items():
        print(f"  {key}: {value}")


if __name__ == "__main__":
    _demonstrate_usage()
