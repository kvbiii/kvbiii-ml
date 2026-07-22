import numpy as np
import pandas as pd
import pytest

from kvbiii_ml.data_processing.preprocessing.categorical_encoding.woe_encoder import (
    WoEEncoderWithOriginal,
)
from kvbiii_ml.data_processing.preprocessing.discretisation.equal_width_discretiser import (
    EqualWidthDiscretiserWithOriginal,
)
from kvbiii_ml.data_processing.preprocessing.outlier_handling.outlier_trimmer import (
    OutlierTrimmerWithOriginal,
)
from kvbiii_ml.data_processing.preprocessing.variance_stabilizing_transformations.log_transformer import (
    LogTransformerWithOriginal,
)

CONTRACT_CASES = [
    pytest.param(
        EqualWidthDiscretiserWithOriginal,
        {"variables": ["numeric_1", "numeric_2"], "bins": 5},
        "numeric_positive_data",
        ["numeric_1", "numeric_2"],
        id="equal_width_discretiser",
    ),
    pytest.param(
        OutlierTrimmerWithOriginal,
        {"variables": ["numeric_1", "numeric_2"]},
        "numeric_positive_data",
        ["numeric_1", "numeric_2"],
        id="outlier_trimmer",
    ),
    pytest.param(
        LogTransformerWithOriginal,
        {"variables": ["numeric_1", "numeric_2"], "base": "e"},
        "numeric_positive_data",
        ["numeric_1", "numeric_2"],
        id="log_transformer",
    ),
    pytest.param(
        WoEEncoderWithOriginal,
        {"variables": ["categorical_1", "categorical_2"]},
        "woe_categorical_data",
        ["categorical_1", "categorical_2"],
        id="woe_encoder",
    ),
]


@pytest.fixture
def numeric_positive_data(test_settings) -> tuple[pd.DataFrame, None]:
    """Provides a strictly-positive numeric DataFrame for unsupervised wrappers.

    Args:
        test_settings: Test configuration fixture.

    Returns:
        tuple[pd.DataFrame, None]: Two-column strictly-positive numeric feature
            matrix paired with a None target for unsupervised wrappers.
    """
    rng = np.random.default_rng(test_settings.SEED)
    X = pd.DataFrame(
        {
            "numeric_1": rng.exponential(scale=5.0, size=test_settings.N_SAMPLES) + 1.0,
            "numeric_2": rng.exponential(scale=3.0, size=test_settings.N_SAMPLES) + 1.0,
        }
    )
    return X, None


@pytest.fixture
def woe_categorical_data() -> tuple[pd.DataFrame, pd.Series]:
    """Provides a categorical DataFrame with a binary target for WoE encoding.

    Every category carries at least one event and one non-event, satisfying
    the Weight of Evidence fitting requirement.

    Returns:
        tuple[pd.DataFrame, pd.Series]: Two-column categorical feature matrix
            and a binary 0/1 target.
    """
    X = pd.DataFrame(
        {
            "categorical_1": [
                "A",
                "A",
                "A",
                "A",
                "B",
                "B",
                "B",
                "B",
                "C",
                "C",
                "C",
                "C",
            ],
            "categorical_2": [
                "X",
                "X",
                "Y",
                "Y",
                "X",
                "X",
                "Y",
                "Y",
                "X",
                "X",
                "Y",
                "Y",
            ],
        }
    )
    y = pd.Series([1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0], name="target")
    return X, y


@pytest.mark.parametrize(
    "estimator_cls, kwargs, dataset_fixture, expected_variables", CONTRACT_CASES
)
def test_with_original_wrapper_fit_returns_self(
    estimator_cls, kwargs, dataset_fixture, expected_variables, request
):
    """Tests that fit() returns the estimator instance across representative wrappers.

    Args:
        estimator_cls: Wrapper class under test.
        kwargs: Constructor keyword arguments for the wrapper.
        dataset_fixture: Name of the fixture providing (X, y) test data.
        expected_variables: Unused here; kept for shared parametrize signature.
        request: Pytest fixture request used to resolve dataset_fixture by name.

    Asserts:
        - fit() returns the same instance it was called on.
    """
    del expected_variables
    X, y = request.getfixturevalue(dataset_fixture)
    estimator = estimator_cls(**kwargs)
    result = estimator.fit(X, y)
    if result is not estimator:
        raise AssertionError(f"fit() did not return self for {estimator_cls.__name__}")


@pytest.mark.parametrize(
    "estimator_cls, kwargs, dataset_fixture, expected_variables", CONTRACT_CASES
)
def test_with_original_wrapper_fit_populates_variables(
    estimator_cls, kwargs, dataset_fixture, expected_variables, request
):
    """Tests that fit() populates variables_ with the configured variable names.

    Args:
        estimator_cls: Wrapper class under test.
        kwargs: Constructor keyword arguments for the wrapper.
        dataset_fixture: Name of the fixture providing (X, y) test data.
        expected_variables: Expected contents of variables_ after fitting.
        request: Pytest fixture request used to resolve dataset_fixture by name.

    Asserts:
        - variables_ matches the configured variable list for every wrapper.
    """
    X, y = request.getfixturevalue(dataset_fixture)
    estimator = estimator_cls(**kwargs)
    estimator.fit(X, y)
    if list(estimator.variables_) != expected_variables:
        raise AssertionError(
            f"variables_ mismatch for {estimator_cls.__name__}: {estimator.variables_}"
        )


@pytest.mark.parametrize(
    "estimator_cls, kwargs, dataset_fixture, expected_variables", CONTRACT_CASES
)
def test_with_original_wrapper_transform_preserves_original_columns(
    estimator_cls, kwargs, dataset_fixture, expected_variables, request
):
    """Tests that transform() leaves the original input columns exactly unchanged.

    Args:
        estimator_cls: Wrapper class under test.
        kwargs: Constructor keyword arguments for the wrapper.
        dataset_fixture: Name of the fixture providing (X, y) test data.
        expected_variables: Variables expected to gain a derived column.
        request: Pytest fixture request used to resolve dataset_fixture by name.

    Asserts:
        - The subset of output columns matching the original input columns is
          identical, values and dtypes included, to the original input.
    """
    del expected_variables
    X, y = request.getfixturevalue(dataset_fixture)
    estimator = estimator_cls(**kwargs)
    estimator.fit(X, y)
    result = estimator.transform(X)
    pd.testing.assert_frame_equal(result[X.columns.tolist()], X)


@pytest.mark.parametrize(
    "estimator_cls, kwargs, dataset_fixture, expected_variables", CONTRACT_CASES
)
def test_with_original_wrapper_transform_appends_suffixed_columns(
    estimator_cls, kwargs, dataset_fixture, expected_variables, request
):
    """Tests that transform() appends one `{variable}_{suffix}` column per variable.

    Args:
        estimator_cls: Wrapper class under test.
        kwargs: Constructor keyword arguments for the wrapper.
        dataset_fixture: Name of the fixture providing (X, y) test data.
        expected_variables: Variables expected to gain a derived column.
        request: Pytest fixture request used to resolve dataset_fixture by name.

    Asserts:
        - Every configured variable gains a `{variable}_{_suffix}` column.
    """
    X, y = request.getfixturevalue(dataset_fixture)
    estimator = estimator_cls(**kwargs)
    estimator.fit(X, y)
    result = estimator.transform(X)
    for variable in expected_variables:
        derived_column = f"{variable}_{estimator._suffix}"
        if derived_column not in result.columns:
            raise AssertionError(
                f"Missing derived column {derived_column} for {estimator_cls.__name__}"
            )


@pytest.mark.parametrize(
    "estimator_cls, kwargs, dataset_fixture, expected_variables", CONTRACT_CASES
)
def test_with_original_wrapper_transform_before_fit_raises_attributeerror(
    estimator_cls, kwargs, dataset_fixture, expected_variables, request
):
    """Tests that transform() before fit() raises AttributeError, not NotFittedError.

    Args:
        estimator_cls: Wrapper class under test.
        kwargs: Constructor keyword arguments for the wrapper.
        dataset_fixture: Name of the fixture providing (X, y) test data.
        expected_variables: Unused here; kept for shared parametrize signature.
        request: Pytest fixture request used to resolve dataset_fixture by name.

    Asserts:
        - Calling transform() on an unfitted wrapper raises AttributeError.
    """
    del expected_variables
    X, _y = request.getfixturevalue(dataset_fixture)
    estimator = estimator_cls(**kwargs)
    with pytest.raises(AttributeError):
        estimator.transform(X)


if __name__ == "__main__":
    print("Run this file with pytest to execute tests.")
