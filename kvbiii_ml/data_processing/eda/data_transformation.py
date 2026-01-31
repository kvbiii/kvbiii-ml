import pandas as pd

from kvbiii_ml.data_processing.feature_engineering.categorical_aligner import (
    CategoricalAligner,
)
from kvbiii_ml.data_processing.feature_engineering.numerical_downcaster import (
    NumericalDowncaster,
)


class DataTransformer:
    """
    Class for transforming data types in a DataFrame to optimize memory usage and prepare data for modeling.
    """

    @staticmethod
    def optimize_memory(
        df: pd.DataFrame, categorical_features: list[str], verbose: bool = True
    ) -> pd.DataFrame:
        """Reduces memory usage by downcasting numerical features and converting specified features to category.

        Args:
            df (pd.DataFrame): Input DataFrame.
            categorical_features (list[str]): List of features to convert to category dtype.
            verbose (bool, optional): Whether to print memory usage before and after optimization. Defaults to True.

        Returns:
            pd.DataFrame: Optimized DataFrame.
        """
        df = df.copy()
        start_mem = df.memory_usage(deep=True).sum() / 1024**2

        numerical_cols = df.select_dtypes(include=["int", "float"]).columns.tolist()
        if numerical_cols:
            if verbose:
                num_start_mem = df.memory_usage(deep=True).sum() / 1024**2
            downcaster = NumericalDowncaster(columns=numerical_cols)
            df = downcaster.fit_transform(df)
            if verbose:
                num_end_mem = df.memory_usage(deep=True).sum() / 1024**2
                print(
                    f"Numerical dtypes reduced: {num_start_mem:.2f} MB → {num_end_mem:.2f} MB ({100*(num_start_mem-num_end_mem)/num_start_mem:.1f}% reduction)"
                )

        if categorical_features:
            if verbose:
                cat_start_mem = df.memory_usage(deep=True).sum() / 1024**2
            aligner = CategoricalAligner(
                categorical_features=categorical_features, warn_on_unknown=False
            )
            df = aligner.fit_transform(df)
            if verbose:
                cat_end_mem = df.memory_usage(deep=True).sum() / 1024**2
                print(
                    f"Categorical dtypes converted: {cat_start_mem:.2f} MB → {cat_end_mem:.2f} MB ({100*(cat_start_mem-cat_end_mem)/cat_start_mem:.1f}% reduction)"
                )

        end_mem = df.memory_usage(deep=True).sum() / 1024**2
        if verbose:
            print(
                f"Total memory usage reduced: {start_mem:.2f} MB → {end_mem:.2f} MB ({100*(start_mem-end_mem)/start_mem:.1f}% reduction)"
            )
        return df

    @staticmethod
    def encode_target_feature(
        df: pd.DataFrame, target_feature: str, verbose: bool = True
    ) -> tuple[pd.DataFrame, dict[int, str]]:
        """Converts target feature to categorical codes and creates label mappings.

        Args:
            df (pd.DataFrame): Input DataFrame containing the target feature.
            target_feature (str): Name of the target feature to process.
            verbose (bool, optional): Whether to print detailed class information. Defaults to True.

        Returns:
            tuple[pd.DataFrame, dict[int, str]]:
                - DataFrame with target feature converted to categorical codes
                - id2label mapping dictionary {id: label}
        """
        if target_feature not in df.columns:
            raise ValueError(
                f"Target feature '{target_feature}' not found in DataFrame"
            )
        df = df.copy()
        df[target_feature] = df[target_feature].astype(str).astype("category")
        id2label = dict(enumerate(df[target_feature].cat.categories))
        if verbose:
            print(f"\nTarget Feature Analysis: '{target_feature}'")
            print(f"• Total unique classes: {len(id2label)}")
            print("• Class mappings:")
            for idx, (id_, label) in enumerate(id2label.items(), 1):
                print(f"  {idx}. Label '{label}' → Encoded as {id_}")
        df[target_feature] = df[target_feature].cat.codes
        return df, id2label


if __name__ == "__main__":
    import numpy as np

    # Create sample data for demonstration
    sample_data = pd.DataFrame(
        {
            "numerical_feature": np.random.randint(0, 1000, 5000),
            "categorical_feature": np.random.choice(["A", "B", "C", "D"], 5000),
            "target_feature": np.random.choice(["yes", "no"], 5000, p=[0.3, 0.7]),
        }
    )

    print("Original DataFrame shape:", sample_data.shape)
    print(
        "Original memory usage:",
        f"{sample_data.memory_usage(deep=True).sum() / 1024**2:.2f} MB",
    )

    transformer = DataTransformer()

    optimized_df = transformer.optimize_memory(
        sample_data, categorical_features=["categorical_feature"]
    )

    print(
        "Optimized memory usage:",
        f"{optimized_df.memory_usage(deep=True).sum() / 1024**2:.2f} MB",
    )

    # 4) Process target feature (encode and return id2label mapping)
    processed_df, id2label = transformer.encode_target_feature(
        optimized_df, target_feature="target_feature", verbose=True
    )

    print("\nEncoded target head:")
    print(processed_df[["target_feature"]].head())
    print("id2label:", id2label)
