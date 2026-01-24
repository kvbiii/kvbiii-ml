import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class CategoriesAssigner(BaseEstimator, TransformerMixin):
    """Assign stable category sets to specified categorical features.

    For each feature in categorical_features, the categories observed during fit
    are captured (preserving order). At transform time, the column is converted
    to a pandas Categorical with the stored categories, ensuring consistent
    category space across datasets. Unknown values (not present in the fitted
    categories) are replaced by the training mode for that feature.
    """

    def __init__(self, categorical_features: list[str]):
        """Initialize the assigner.

        Args:
            categorical_features (list[str]): List of column names expected to be
                categorical and whose category levels should be fixed after fit.
        """
        self.categorical_features = categorical_features

    def fit(self, X: pd.DataFrame, y: object | None = None) -> "CategoriesAssigner":
        """Extract category levels and modes for configured features.

        Each listed feature must already be a pandas Categorical in X; its
        category levels are stored for later enforcement during transform.
        The most frequent (mode) non-null value is saved for unknown replacement.

        Args:
            X (pd.DataFrame): Input DataFrame.
            y (object | None, optional): Ignored. Present for compatibility.

        Returns:
            CategoriesAssigner: Fitted transformer.
        """
        self.input_features_ = X.columns
        self.feature_groups = {
            feature: X[feature].cat.categories.tolist()
            for feature in self.categorical_features
            if feature in X.columns
            and isinstance(X[feature].dtype, pd.CategoricalDtype)
        }
        self.feature_modes_ = {}
        for feature, categories in self.feature_groups.items():
            series = X[feature]
            non_null = series.dropna()
            if not non_null.empty:
                mode_val = non_null.mode().iloc[0]
            else:
                mode_val = categories[0] if categories else None
            self.feature_modes_[feature] = mode_val
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Apply stored category definitions and replace unknowns with mode."""
        X = X.copy()
        for feature, categories in self.feature_groups.items():
            if feature in X.columns:
                mode_val = self.feature_modes_.get(feature)
                s = X[feature].copy().astype("string")
                not_na = s.notna()
                categories_str = [str(c) for c in categories]
                mask_unknown = not_na & ~s.isin(categories_str)
                if (
                    mask_unknown.any()
                    and mode_val is not None
                    and not pd.isna(mode_val)
                ):
                    s.loc[mask_unknown] = str(mode_val)
                X[feature] = pd.Categorical(s, categories=categories_str)
        return X

    def get_feature_names_out(
        self, input_features: list[str] | None = None
    ) -> pd.Index:
        """Return feature names seen during fit.

        Args:
            input_features (list[str] | None, optional): Unused; for API symmetry.

        Returns:
            pd.Index: Original feature names.
        """
        return self.input_features_


if __name__ == "__main__":
    # Minimal example
    df_train = pd.DataFrame(
        {
            "color": pd.Categorical(["red", "blue", "green", "red"]),
            "fuel": pd.Categorical(["petrol", "diesel", "electric", "diesel"]),
            "mileage": [10000, 15000, 12000, 18000],
        }
    )
    df_test = pd.DataFrame(
        {
            "color": ["blue", "purple", "yellow"],  # purple, yellow unseen
            "fuel": ["diesel", "hydrogen", "petrol"],  # hydrogen unseen
            "mileage": [13000, 17000, 9000],
        }
    )

    assigner = CategoriesAssigner(["color", "fuel"]).fit(df_train)
    transformed = assigner.transform(df_test)
    print(transformed)
    print(transformed.dtypes)
    print(df_train["color"].unique())
    print(transformed["color"].unique())
