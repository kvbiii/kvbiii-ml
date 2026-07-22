import numpy as np
import pandas as pd
import pytest

from kvbiii_ml.data_processing.preprocessing.discretisation.decision_tree_discretiser import (
    DecisionTreeDiscretiserWithOriginal,
)

CV_FOLDS = 2
N_ROWS = 60


@pytest.fixture
def decision_tree_regression_data() -> tuple[pd.DataFrame, pd.Series]:
    """Provides a small regression dataset for decision-tree discretisation.

    Returns:
        tuple[pd.DataFrame, pd.Series]: Feature matrix with `age` and `income`
            columns paired with a continuous target.
    """
    rng = np.random.default_rng(42)
    X = pd.DataFrame(
        {
            "age": rng.uniform(0, 100, N_ROWS),
            "income": rng.uniform(0, 100, N_ROWS),
        }
    )
    y = pd.Series(rng.uniform(0, 1, N_ROWS), name="target")
    return X, y


def test_decisiontreediscretiserwithoriginal_fit_populates_variables(
    decision_tree_regression_data,
):
    """Tests that fit populates variables_ with the configured columns.

    Args:
        decision_tree_regression_data: Feature matrix and target fixture.

    Asserts:
        - variables_ matches the configured variable list.
    """
    X, y = decision_tree_regression_data
    discretiser = DecisionTreeDiscretiserWithOriginal(
        variables=["age"], cv=CV_FOLDS, regression=True, random_state=42
    )
    discretiser.fit(X, y)
    if list(discretiser.variables_) != ["age"]:
        raise AssertionError(f"Unexpected variables_: {discretiser.variables_}")


def test_decisiontreediscretiserwithoriginal_transform_preserves_original_columns(
    decision_tree_regression_data,
):
    """Tests that transform preserves the original columns exactly.

    Args:
        decision_tree_regression_data: Feature matrix and target fixture.

    Asserts:
        - The original columns of the output match the input exactly.
    """
    X, y = decision_tree_regression_data
    discretiser = DecisionTreeDiscretiserWithOriginal(
        variables=["age"], cv=CV_FOLDS, regression=True, random_state=42
    )
    discretiser.fit(X, y)
    result = discretiser.transform(X)
    pd.testing.assert_frame_equal(result[X.columns.tolist()], X)


@pytest.mark.parametrize("bin_output", ["prediction", "bin_number"])
def test_decisiontreediscretiserwithoriginal_transform_respects_bin_output(
    decision_tree_regression_data, bin_output
):
    """Tests that `bin_output` controls whether derived values are predictions or bin codes.

    Args:
        decision_tree_regression_data: Feature matrix and target fixture.
        bin_output: Configured `bin_output` value under test.

    Asserts:
        - The derived column is produced without error for both allowed values.
        - `bin_number` output only contains non-negative integer-like codes.
    """
    X, y = decision_tree_regression_data
    discretiser = DecisionTreeDiscretiserWithOriginal(
        variables=["age"],
        cv=CV_FOLDS,
        regression=True,
        random_state=42,
        bin_output=bin_output,
    )
    discretiser.fit(X, y)
    result = discretiser.transform(X)
    derived = result["age_PREPROCESS_DT_DISC"]
    if derived.isna().any():
        raise AssertionError(f"Derived column contains NaN for bin_output={bin_output}.")
    if bin_output == "bin_number":
        if not (derived >= 0).all():
            raise AssertionError("bin_number output should only contain non-negative codes.")


if __name__ == "__main__":
    print("Run this file with pytest to execute tests.")
