import pandas as pd
from imblearn.over_sampling import SMOTENC, RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler


class FoldwiseSampler:
    """
    A wrapper class for applying various sampling strategies consistently across CV folds.
    """

    def __init__(self, strategy="none", sampler_params=None) -> None:
        """
        Initialize the FoldwiseSampler with a specified strategy and parameters.

        Args:
            strategy (str): Sampling strategy to use. Options are "none", "smote",
                "random_over", "random_under". Default is "none".
            sampler_params (dict, optional): Additional parameters for the sampler.
        """
        self.strategy = strategy
        self.sampler_params = sampler_params or {}
        self.sampler = None

    def fit(self, X: pd.DataFrame, y: pd.Series) -> None:
        """
        Initialize the sampler based on the chosen strategy.

        Args:
            X (pd.DataFrame): Feature matrix.
            y (pd.Series): Target vector.
        """
        if self.strategy == "none":
            self.sampler = None
        elif self.strategy == "smote":
            self.sampler = SMOTENC(**self.sampler_params)
        elif self.strategy == "random_over":
            self.sampler = RandomOverSampler(**self.sampler_params)
        elif self.strategy == "random_under":
            self.sampler = RandomUnderSampler(**self.sampler_params)
        else:
            raise ValueError(f"Unknown sampling strategy: {self.strategy}")

    def fit_resample(
        self, X: pd.DataFrame, y: pd.Series
    ) -> tuple[pd.DataFrame, pd.Series]:
        """
        Apply the sampling strategy to the data.

        Args:
            X (pd.DataFrame): Feature matrix.
            y (pd.Series): Target vector.
        
        Returns:
            tuple[pd.DataFrame, pd.Series]: Resampled feature matrix and target vector.
        """
        if self.sampler is None:
            return X, y
        X_resampled, y_resampled = self.sampler.fit_resample(X, y)
        X_resampled = pd.DataFrame(X_resampled, columns=X.columns)
        y_resampled = pd.Series(y_resampled, name=y.name)
        return X_resampled, y_resampled

    def fit_transform(
        self, X: pd.DataFrame, y: pd.Series
    ) -> tuple[pd.DataFrame, pd.Series]:
        """
        Fit and apply the sampling strategy to the data.

        Args:
            X (pd.DataFrame): Feature matrix.
            y (pd.Series): Target vector.
        
        Returns:
            tuple[pd.DataFrame, pd.Series]: Resampled feature matrix and target vector.
        """
        self.fit(X, y)
        return self.fit_resample(X, y)
