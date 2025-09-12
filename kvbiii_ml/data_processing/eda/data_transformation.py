import pandas as pd


class DataTransformer:
    """
    Class for transforming data types in a DataFrame to optimize memory usage and prepare data for modeling.
    """

    @staticmethod
    def reduce_numerical_dtypes(df: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
        """Downcasts numerical features to more efficient types where possible.

        Args:
            df (pd.DataFrame): Input DataFrame with numerical features.
            verbose (bool, optional): Whether to print memory usage before and after
                reduction. Defaults to True.

        Returns:
            pd.DataFrame: DataFrame with reduced numerical dtypes.
        """
        df = df.copy()
        start_mem = df.memory_usage(deep=True).sum() / 1024**2
        for col in df.select_dtypes(include=["int", "float"]).columns:
            col_type = df[col].dtype
            if pd.api.types.is_integer_dtype(col_type):
                df[col] = pd.to_numeric(df[col], downcast="integer")
            elif pd.api.types.is_float_dtype(col_type):
                df[col] = pd.to_numeric(df[col], downcast="float")
        end_mem = df.memory_usage(deep=True).sum() / 1024**2
        if verbose:
            print(
                f"Numerical dtypes reduced: {start_mem:.2f} MB → {end_mem:.2f} MB ({100*(start_mem-end_mem)/start_mem:.1f}% reduction)"
            )
        return df

    @staticmethod
    def convert_to_category(
        df: pd.DataFrame, categorical_features: list[str], verbose: bool = True
    ) -> pd.DataFrame:
        """Converts specified features to category dtype.

        Args:
            df (pd.DataFrame): Input DataFrame.
            categorical_features (list[str]): List of features to convert to category dtype.
            verbose (bool, optional): Whether to print memory usage before and after
                conversion. Defaults to True.

        Returns:
            pd.DataFrame: DataFrame with specified features converted to category dtype.
        """
        df = df.copy()
        start_mem = df.memory_usage(deep=True).sum() / 1024**2
        for col in categorical_features:
            if col in df.columns:
                df[col] = df[col].astype(str)
                df[col] = df[col].astype("category")
        end_mem = df.memory_usage(deep=True).sum() / 1024**2
        if verbose:
            print(
                f"Categorical dtypes converted: {start_mem:.2f} MB → {end_mem:.2f} MB ({100*(start_mem-end_mem)/start_mem:.1f}% reduction)"
            )
        return df

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
        df = DataTransformer.reduce_numerical_dtypes(df, verbose=verbose)
        df = DataTransformer.convert_to_category(
            df, categorical_features, verbose=verbose
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

    # 1) Reduce numerical dtypes
    reduced_df = transformer.reduce_numerical_dtypes(sample_data)

    # 2) Convert selected features to category
    cat_feats = ["categorical_feature"]
    cat_df = transformer.convert_to_category(reduced_df, cat_feats)

    # 3) Full optimization (reduce + convert)
    optimized_df = transformer.optimize_memory(
        sample_data, categorical_features=cat_feats
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
