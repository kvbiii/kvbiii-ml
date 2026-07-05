from typing import Any

import pandas as pd
from feature_engine.discretisation import DecisionTreeDiscretiser

from kvbiii_ml.data_processing.preprocessing.expansion_base import _WithOriginalBase


class DecisionTreeDiscretiserWithOriginal(_WithOriginalBase):
    """Wraps DecisionTreeDiscretiser to keep originals and append tree-binned copies.

    Derived columns are named ``{original}_PREPROCESS_DT_DISC``. Each variable is discretised
    using a decision tree fitted against the target; leaf predictions become the
    encoded value.
    """

    _suffix = "PREPROCESS_DT_DISC"

    def __init__(
        self,
        variables: list[str] | None = None,
        bin_output: str = "prediction",
        precision: int | None = None,
        cv: int = 3,
        scoring: str = "neg_mean_squared_error",
        param_grid: dict | None = None,
        regression: bool = True,
        random_state: int | None = None,
    ) -> None:
        """
        Initialize DecisionTreeDiscretiserWithOriginal.

        Args:
            variables (list[str] | None, optional): Numerical columns to discretise.
                Defaults to None (auto-detect).
            bin_output (str, optional): What the tree leaf value represents -
                ``"prediction"`` or ``"bin_number"``. Defaults to ``"prediction"``.
            precision (int | None, optional): Decimal places for rounding. Defaults to None.
            cv (int, optional): Cross-validation folds used for tree depth tuning.
                Defaults to 3.
            scoring (str, optional): sklearn scoring metric for CV. Defaults to
                ``"neg_mean_squared_error"``.
            param_grid (dict | None, optional): Grid of tree parameters to search.
                Defaults to None.
            regression (bool, optional): If True, fits a regression tree; if False,
                a classification tree. Defaults to True.
            random_state (int | None, optional): Random seed. Defaults to None.
        """
        self.variables = variables
        self.bin_output = bin_output
        self.precision = precision
        self.cv = cv
        self.scoring = scoring
        self.param_grid = param_grid
        self.regression = regression
        self.random_state = random_state

    def _build_inner(self) -> DecisionTreeDiscretiser:
        """Return a fresh DecisionTreeDiscretiser.

        Returns:
            DecisionTreeDiscretiser: Unfitted instance.
        """
        return DecisionTreeDiscretiser(
            variables=self.variables,
            bin_output=self.bin_output,
            precision=self.precision,
            cv=self.cv,
            scoring=self.scoring,
            param_grid=self.param_grid,
            regression=self.regression,
            random_state=self.random_state,
        )

    def _fit_inner(
        self, inner: DecisionTreeDiscretiser, X: pd.DataFrame, y: Any
    ) -> DecisionTreeDiscretiser:
        """Fit the discretiser on X and y.

        Args:
            inner (DecisionTreeDiscretiser): Unfitted discretiser.
            X (pd.DataFrame): Training features.
            y (Any): Target used to fit the per-variable decision trees.

        Returns:
            DecisionTreeDiscretiser: Fitted discretiser.
        """
        return inner.fit(X, y)

    def _transform_inner(
        self, inner: DecisionTreeDiscretiser, X: pd.DataFrame
    ) -> pd.DataFrame:
        """Apply the fitted discretiser to X.

        Args:
            inner (DecisionTreeDiscretiser): Fitted discretiser.
            X (pd.DataFrame): Features to transform.

        Returns:
            pd.DataFrame: DataFrame with tree-binned column values.
        """
        return inner.transform(X)


__all__ = ["DecisionTreeDiscretiser", "DecisionTreeDiscretiserWithOriginal"]


if __name__ == "__main__":
    import numpy as np
    import pandas as pd

    rng = np.random.default_rng(42)
    n = 300
    df = pd.DataFrame(
        {
            "age": rng.integers(18, 80, n).astype(float),
            "income": rng.exponential(50_000, n),
        }
    )
    y = pd.Series((df["income"] > 50_000).astype(int), name="target")

    enc = DecisionTreeDiscretiser(variables=["age", "income"], regression=False, cv=3)
    print("=== DecisionTreeDiscretiser (replace) ===")
    print(enc.fit_transform(df, y).head())

    enc_exp = DecisionTreeDiscretiserWithOriginal(
        variables=["age", "income"], regression=False, cv=3
    )
    print("\n=== DecisionTreeDiscretiserWithOriginal (append) ===")
    print(enc_exp.fit_transform(df, y).head())
