import pandas as pd


class DataCleaner:
    """A comprehensive data cleaning class for preprocessing DataFrames.

    This class provides methods for cleaning and preprocessing data including
    removing unnecessary features, handling missing values, optimizing data types,
    and identifying categorical features.
    """

    @staticmethod
    def get_categorical_features(df: pd.DataFrame, threshold: int = 100) -> list[str]:
        """Gets categorical features from the DataFrame.

        Args:
            df (pd.DataFrame): The input DataFrame.
            threshold (int, optional): Maximum number of unique values for a feature
                to be considered categorical. Defaults to 100.

        Returns:
            list[str]: List of categorical feature names.
        """
        categorical_features = df.select_dtypes(include=["object", "category"]).columns
        categorical_features = [
            col for col in categorical_features if df[col].nunique() < threshold
        ]
        categorical_features = list(
            set(categorical_features) | set(df.columns[df.nunique() == 2])
        )
        return categorical_features

    @staticmethod
    def initial_features_removal(df: pd.DataFrame) -> pd.DataFrame:
        """Removes features that are not useful for modeling.

        Removes features with:
        - Only 1 unique value (variance = 0)
        - All missing values
        - Duplicated features

        Args:
            df (pd.DataFrame): DataFrame to be cleaned.

        Returns:
            pd.DataFrame: Cleaned DataFrame with unnecessary features removed.
        """
        number_of_features_before = df.shape[1]

        # Remove features with only 1 unique value (if there are no nulls)
        nunique_mask = df.nunique(dropna=False) > 1
        single_unique_features = df.columns[~nunique_mask].tolist()
        if single_unique_features:
            print(
                f"Removed features with only 1 unique value: {single_unique_features}"
            )
        df = df.loc[:, nunique_mask]

        # Remove features with all missing values
        na_mask = ~df.isna().all()
        all_na_features = df.columns[~na_mask].tolist()
        if all_na_features:
            print(f"Removed features with all missing values: {all_na_features}")
        df = df.loc[:, na_mask]

        # Remove duplicated features
        feats = list(df.columns)
        to_drop = set()
        for i in range(len(feats)):
            feat1 = feats[i]
            if feat1 in to_drop:
                continue
            for j in range(i + 1, len(feats)):
                feat2 = feats[j]
                if feat2 in to_drop:
                    continue
                if df[feat1].equals(df[feat2]):
                    print(
                        f"Features {feat1} and {feat2} are identical. Removing the feature: {feat2}"
                    )
                    to_drop.add(feat2)
        if to_drop:
            df = df.drop(columns=list(to_drop))

        print(
            f"\nNumber of features after removal: {df.shape[1]}. Removed {number_of_features_before - df.shape[1]} features."
        )
        return df.reset_index(drop=True)

    @staticmethod
    def drop_highly_skewed_categorical_features(
        df: pd.DataFrame, features_to_investigate: list[str], threshold: float = 0.99
    ) -> pd.DataFrame:
        """Drops categorical features with highly skewed distributions.

        Removes features where one value accounts for more than the specified
        threshold of the data.

        Args:
            df (pd.DataFrame): The input DataFrame.
            features_to_investigate (list[str]): List of categorical features to investigate.
            threshold (float, optional): The threshold for skewness in the distribution.
                Defaults to 0.99.

        Returns:
            pd.DataFrame: The DataFrame with highly skewed categorical features dropped.
        """
        df = df.copy()
        feats_to_drop = []
        for feature in features_to_investigate:
            value_counts = df[feature].value_counts(normalize=True, dropna=False)
            if value_counts.max() > threshold:
                print(
                    f"  - {feature} (value '{value_counts.idxmax()}' with {value_counts.max()*100:.2f}% of the data)"
                )
                feats_to_drop.append(feature)
        if not feats_to_drop:
            return df
        print(
            f"Highly skewed categorical features with a distribution where one value is more than {threshold*100:.2f}% of the data:\n"
        )
        df = df.drop(columns=feats_to_drop)
        print(
            f"\nNumber of features after removal: {df.shape[1]}. Removed {len(feats_to_drop)} features."
        )
        return df

    @staticmethod
    def categorize_categorical_features_by_missing(
        data: pd.DataFrame, categorical_features: list[str]
    ) -> dict[str, list[str]]:
        """Categorizes categorical features by their proportion of missing values.

        Args:
            data (pd.DataFrame): The input DataFrame.
            categorical_features (list[str]): List of categorical features.

        Returns:
            dict[str, list[str]]: Dictionary with keys as types of categorical features
                and values as lists of feature names.
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


if __name__ == "__main__":
    import numpy as np

    # Create sample data for demonstration
    sample_data = pd.DataFrame(
        {
            "numerical_feature": np.random.randint(0, 100, 1000),
            "categorical_feature": np.random.choice(["A", "B", "C", "D"], 1000),
            "duplicate_feature": np.random.randint(0, 100, 1000),
            "single_value_feature": ["constant"] * 1000,
            "all_na_feature": [np.nan] * 1000,
            "high_cardinality_feature": np.random.choice(range(500), 1000),
            "skewed_cat_feature": np.random.choice(
                ["majority", "minority"], 1000, p=[0.995, 0.005]
            ),
        }
    )

    # Make duplicate feature identical to numerical_feature
    sample_data["duplicate_feature"] = sample_data["numerical_feature"].copy()

    # Add some missing values
    sample_data.loc[50:60, "categorical_feature"] = np.nan

    print("Original DataFrame shape:", sample_data.shape)
    print(
        "Original memory usage:",
        f"{sample_data.memory_usage(deep=True).sum() / 1024**2:.2f} MB",
    )

    # Demonstrate DataCleaner usage
    cleaner = DataCleaner()

    # Initial feature removal
    cleaned_data = cleaner.initial_features_removal(sample_data)

    # Get categorical features
    cat_feats = cleaner.get_categorical_features(cleaned_data, threshold=50)
    print(f"\nCategorical features (threshold=50): {cat_feats}")

    # Drop highly skewed categorical features
    cleaned_data = cleaner.drop_highly_skewed_categorical_features(
        cleaned_data, cat_feats, threshold=0.99
    )

    # Update categorical features after dropping skewed ones
    cat_feats = cleaner.get_categorical_features(cleaned_data, threshold=50)

    # Categorize categorical features by missing values
    cat_categories = cleaner.categorize_categorical_features_by_missing(
        cleaned_data, cat_feats
    )
    print(f"\nCategorical features categorized by missing values:")
    for key, value in cat_categories.items():
        print(f"  {key}: {value}")
