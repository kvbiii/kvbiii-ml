import inspect
import pandas as pd
from typing import Callable
import copy


class FeatureGenerationPipeline:
    """
    Class for managing a feature generation pipeline.
    """

    def __init__(
        self,
        features_names: list[str] | None = None,
        preprocessing_steps: list | None = None,
        custom_features_generation_before_preprocessing: (
            Callable[[pd.DataFrame], pd.DataFrame] | None
        ) = None,
        custom_features_generation_after_preprocessing: (
            Callable[[pd.DataFrame], pd.DataFrame] | None
        ) = None,
    ) -> None:
        """
        Initialize the FeatureGenerationPipeline class.

        Args:
            features_names (list[str]): List of feature names to be processed.
            preprocessing_steps (list): List of preprocessing steps to apply.
            custom_features_generation_before_preprocessing (callable, optional): Function to add custom features before preprocessing.
                Should accept a pd.DataFrame and return a pd.DataFrame.
            custom_features_generation_after_preprocessing (callable, optional): Function to add custom features after preprocessing.
                Should accept a pd.DataFrame and return a pd.DataFrame.
        """
        self.features_names = features_names if features_names is not None else []
        self.preprocessing_steps = (
            preprocessing_steps if preprocessing_steps is not None else []
        )

        if (
            custom_features_generation_before_preprocessing is not None
            and not callable(custom_features_generation_before_preprocessing)
        ):
            raise ValueError(
                "custom_features_generation_before_preprocessing must be callable or None."
            )
        if custom_features_generation_after_preprocessing is not None and not callable(
            custom_features_generation_after_preprocessing
        ):
            raise ValueError(
                "custom_features_generation_after_preprocessing must be callable or None."
            )

        self.custom_features_generation_before_preprocessing = (
            custom_features_generation_before_preprocessing or self._identity
        )
        self.custom_features_generation_after_preprocessing = (
            custom_features_generation_after_preprocessing or self._identity
        )

    @staticmethod
    def _identity(df: pd.DataFrame) -> pd.DataFrame:
        """
        Default no-op custom feature function.

        Args:
            df (pd.DataFrame): DataFrame to return unchanged.

        Returns:
            pd.DataFrame: The input DataFrame unchanged.
        """
        return df

    def fit(self, X: pd.DataFrame, y: pd.Series = None) -> "FeatureGenerationPipeline":
        """
        Fit the feature generation process on the DataFrame.

        Args:
            X (pd.DataFrame): DataFrame to fit.
            y (pd.Series, optional): Target variable. Defaults to None.

        Returns:
            FeatureGenerationPipeline: Fitted instance.
        """
        X = self.custom_features_generation_before_preprocessing(copy.deepcopy(X))

        base_cols = set(X.columns)
        cross_new_cols: list[str] | None = None

        for step in self.preprocessing_steps:
            # If a step requests dynamically generated features, assign them now
            if hasattr(step, "features_names"):
                fn = getattr(step, "features_names")
                wants_dynamic = (isinstance(fn, str) and fn == "__NEW__") or (
                    isinstance(fn, list) and len(fn) == 0
                )
                if wants_dynamic:
                    if cross_new_cols is not None:
                        step.features_names = list(cross_new_cols)
                    else:
                        step.features_names = [
                            c for c in X.columns if c not in base_cols
                        ]

            fit = getattr(step, "fit", None)
            if fit is None:
                raise ValueError("All preprocessing steps must implement a fit method.")
            # Efficiently call fit with y only if it accepts it
            fit_params = inspect.signature(fit).parameters
            if "y" in fit_params and y is not None:
                fit(X, y)
            else:
                fit(X)

            # Propagate transformed features forward so later steps can see new columns
            transform = getattr(step, "transform", None)
            if callable(transform):
                X = transform(X)
                # If this step is CrossFeatureGenerator, remember its new columns
                if step.__class__.__name__ == "CrossFeatureGenerator":
                    cross_new_cols = [c for c in X.columns if c not in base_cols]
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Transform the DataFrame with the fitted feature generation process.

        Args:
            X (pd.DataFrame): DataFrame to transform.

        Returns:
            pd.DataFrame: Transformed DataFrame with generated features.
        """
        X = self.custom_features_generation_before_preprocessing(X)
        for step in self.preprocessing_steps:
            X = step.transform(X)
        X = self.custom_features_generation_after_preprocessing(X)
        return X

    def fit_transform(self, X: pd.DataFrame, y: pd.Series = None) -> pd.DataFrame:
        """
        Fit and transform the DataFrame with the feature generation process.

        Args:
            X (pd.DataFrame): DataFrame to fit and transform.
            y (pd.Series, optional): Target variable. Defaults to None.

        Returns:
            pd.DataFrame: Transformed DataFrame with generated features.
        """
        self.fit(X, y)
        return self.transform(X)


if __name__ == "__main__":
    # Minimal runnable example to sanity-check the pipeline
    from sklearn.model_selection import KFold
    from kvbiii_ml.data_processing.feature_engineering.count_encoding import (
        CountEncodingFeatureGenerator,
    )
    from kvbiii_ml.data_processing.feature_engineering.cross_encoding import (
        CrossFeatureGenerator,
    )
    from kvbiii_ml.data_processing.feature_engineering.target_encoding import (
        TargetEncodingFeatureGenerator,
    )

    # Toy data
    data = {
        "A": ["a", "b", "a", "c", "b", "a", "c", "b", "a", "c"],
        "B": ["x", "y", "x", "z", "y", "z", "x", "y", "z", "x"],
    }
    X = pd.DataFrame(data)
    y = pd.Series([1, 0, 1, 0, 1, 0, 1, 0, 1, 0], name="target")

    steps = [
        CrossFeatureGenerator(features_names=[], degree=2),
        TargetEncodingFeatureGenerator(
            features_names=[],
            aggregation="mean",
            smooth=10,
            cv=KFold(n_splits=5, shuffle=True, random_state=17),
        ),
        CountEncodingFeatureGenerator(features_names=[]),
    ]

    def custom_feature_generation_before(df: pd.DataFrame) -> pd.DataFrame:
        df["A_squared"] = df["A"].apply(lambda x: x + "_squared")
        return df

    pipe = FeatureGenerationPipeline(
        features_names=["A", "B"],
        preprocessing_steps=steps,
        custom_features_generation_before_preprocessing=custom_feature_generation_before,
    )

    X_enc = pipe.fit_transform(X, y)
    print("Input columns:", list(X.columns))
    print("Output columns:", list(X_enc.columns))
    print("\nEncoded head:\n", X_enc.head())
