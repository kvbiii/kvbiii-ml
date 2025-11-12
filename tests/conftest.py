"""Centralized test configuration and fixtures for kvbiii-ml package."""

import pytest
import numpy as np
import pandas as pd
from unittest.mock import Mock, MagicMock
from sklearn.base import BaseEstimator, TransformerMixin, ClassifierMixin
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import KFold
from pydantic_settings import BaseSettings, SettingsConfigDict


class TestSettings(BaseSettings):
    """Centralized test environment configuration.

    Attributes:
        SEED (int): Seed for reproducibility across all tests
        N_SAMPLES (int): Default number of samples for test datasets
        N_FEATURES (int): Default number of features for test datasets
        TEST_SIZE (float): Default test split ratio
    """

    SEED: int = 17
    N_SAMPLES: int = 100
    N_FEATURES: int = 5
    TEST_SIZE: float = 0.2

    model_config = SettingsConfigDict(env_file=".env.test", frozen=True, extra="forbid")


@pytest.fixture(scope="session")
def test_settings() -> TestSettings:
    """Provides the test settings configuration.
    
    Returns:
        TestSettings: Loaded configuration object
        
    Raises:
        pytest.Exception: On invalid/missing configuration
    """
    try:
        return TestSettings()
    except Exception as e:
        pytest.fail(f"Test configuration failed: {str(e)}")


@pytest.fixture
def sample_dataframe(test_settings: TestSettings) -> pd.DataFrame:
    """Provides a sample DataFrame for testing purposes.

    Args:
        test_settings (TestSettings): Test settings fixture.
    
    Returns:
        pd.DataFrame: Sample DataFrame with mixed data types for comprehensive testing.
    """
    np.random.seed(test_settings.SEED)
    return pd.DataFrame({
        'numeric_1': np.random.rand(test_settings.N_SAMPLES),
        'numeric_2': np.random.rand(test_settings.N_SAMPLES),
        'categorical_1': np.random.choice(['A', 'B', 'C'], size=test_settings.N_SAMPLES),
        'categorical_2': np.random.choice(['X', 'Y'], size=test_settings.N_SAMPLES),
        'integer_1': np.random.randint(0, 10, size=test_settings.N_SAMPLES)
    })


@pytest.fixture
def sample_series(test_settings: TestSettings) -> pd.Series:
    """Provides a sample Series for testing purposes.

    Args:
        test_settings (TestSettings): Test settings fixture.
    
    Returns:
        pd.Series: Sample Series with random numeric data.
    """
    np.random.seed(test_settings.SEED)
    return pd.Series(np.random.rand(test_settings.N_SAMPLES), name='test_series')


@pytest.fixture
def binary_classification_data(test_settings: TestSettings) -> tuple[pd.DataFrame, pd.Series]:
    """Provides binary classification dataset for testing.

    Args:
        test_settings (TestSettings): Test settings fixture.
    
    Returns:
        tuple[pd.DataFrame, pd.Series]: Feature matrix and binary target vector.
    """
    np.random.seed(test_settings.SEED)
    X = pd.DataFrame(
        np.random.randn(test_settings.N_SAMPLES, test_settings.N_FEATURES),
        columns=[f'feature_{i}' for i in range(test_settings.N_FEATURES)]
    )
    y = pd.Series(
        (np.random.rand(test_settings.N_SAMPLES) > 0.5).astype(int),
        name='target'
    )
    return X, y


@pytest.fixture
def multiclass_classification_data(test_settings: TestSettings) -> tuple[pd.DataFrame, pd.Series]:
    """Provides multiclass classification dataset for testing.

    Args:
        test_settings (TestSettings): Test settings fixture.
    
    Returns:
        tuple[pd.DataFrame, pd.Series]: Feature matrix and multiclass target vector.
    """
    np.random.seed(test_settings.SEED)
    X = pd.DataFrame(
        np.random.randn(test_settings.N_SAMPLES, test_settings.N_FEATURES),
        columns=[f'feature_{i}' for i in range(test_settings.N_FEATURES)]
    )
    y = pd.Series(
        np.random.choice([0, 1, 2], size=test_settings.N_SAMPLES),
        name='target'
    )
    return X, y


@pytest.fixture
def regression_data(test_settings: TestSettings) -> tuple[pd.DataFrame, pd.Series]:
    """Provides regression dataset for testing.

    Args:
        test_settings (TestSettings): Test settings fixture.
    
    Returns:
        tuple[pd.DataFrame, pd.Series]: Feature matrix and continuous target vector.
    """
    np.random.seed(test_settings.SEED)
    X = pd.DataFrame(
        np.random.randn(test_settings.N_SAMPLES, test_settings.N_FEATURES),
        columns=[f'feature_{i}' for i in range(test_settings.N_FEATURES)]
    )
    y = pd.Series(
        np.random.randn(test_settings.N_SAMPLES),
        name='target'
    )
    return X, y


@pytest.fixture
def mock_estimator() -> Mock:
    """Provides a mock sklearn estimator for testing.
    
    Returns:
        Mock: Mock estimator with predict and predict_proba methods configured.
    """
    estimator = Mock()
    estimator.fit.return_value = estimator
    estimator.predict.return_value = np.array([0, 1, 0, 1, 1])
    estimator.predict_proba.return_value = np.array([[0.8, 0.2], [0.3, 0.7], [0.9, 0.1], [0.4, 0.6], [0.2, 0.8]])
    estimator.feature_names_in_ = np.array(['feature_0', 'feature_1', 'feature_2', 'feature_3', 'feature_4'])
    
    # Configure fit method to accept various parameters
    def mock_fit(X, y, **kwargs):
        return estimator
    estimator.fit = Mock(side_effect=mock_fit)
    
    return estimator


@pytest.fixture
def mock_transformer() -> Mock:
    """Provides a mock sklearn transformer for testing.
    
    Returns:
        Mock: Mock transformer with fit, transform, and fit_transform methods configured.
    """
    transformer = Mock()  # Remove spec=TransformerMixin restriction
    transformer.fit.return_value = transformer
    transformer.transform.return_value = pd.DataFrame({'transformed': [1, 2, 3, 4, 5]})
    transformer.fit_transform.return_value = pd.DataFrame({'transformed': [1, 2, 3, 4, 5]})
    transformer.fit_resample.return_value = (pd.DataFrame({'resampled': [1, 2, 3]}), pd.Series([0, 1, 0]))
    return transformer


@pytest.fixture 
def mock_processor() -> Mock:
    """Provides a mock processor with fit_resample method for testing.
    
    Returns:
        Mock: Mock processor with fit_resample method configured.
    """
    processor = Mock()
    # Return data that maintains the same feature structure
    def mock_fit_resample(X, y):
        return X.copy(), y.copy()  # Return the same data structure
    
    processor.fit_resample = Mock(side_effect=mock_fit_resample)
    return processor


@pytest.fixture
def logistic_regression_estimator(test_settings: TestSettings) -> LogisticRegression:
    """Provides a configured LogisticRegression estimator for testing.

    Args:
        test_settings (TestSettings): Test settings fixture.
    
    Returns:
        LogisticRegression: Configured estimator with reproducible random state.
    """
    return LogisticRegression(
        max_iter=200, 
        random_state=test_settings.SEED,
        solver='liblinear'
    )


@pytest.fixture
def linear_regression_estimator(test_settings: TestSettings) -> LinearRegression:
    """Provides a configured LinearRegression estimator for testing.

    Args:
        test_settings (TestSettings): Test settings fixture.
    
    Returns:
        LinearRegression: Configured regression estimator.
    """
    return LinearRegression()


@pytest.fixture
def random_forest_estimator(test_settings: TestSettings) -> RandomForestClassifier:
    """Provides a configured RandomForestClassifier estimator for testing.

    Args:
        test_settings (TestSettings): Test settings fixture.
    
    Returns:
        RandomForestClassifier: Configured estimator with reproducible random state.
    """
    return RandomForestClassifier(
        n_estimators=10,
        random_state=test_settings.SEED,
        max_depth=3
    )


@pytest.fixture
def kfold_cv(test_settings: TestSettings) -> KFold:
    """Provides a configured KFold cross-validator for testing.

    Args:
        test_settings (TestSettings): Test settings fixture.
    
    Returns:
        KFold: Configured cross-validator with reproducible random state.
    """
    return KFold(n_splits=3, shuffle=True, random_state=test_settings.SEED)


@pytest.fixture
def sample_predictions(test_settings: TestSettings) -> tuple[np.ndarray, np.ndarray]:
    """Provides sample predictions for evaluation testing.

    Args:
        test_settings (TestSettings): Test settings fixture.
    
    Returns:
        tuple[np.ndarray, np.ndarray]: True labels and predicted labels.
    """
    np.random.seed(test_settings.SEED)
    y_true = np.random.choice([0, 1], size=test_settings.N_SAMPLES)
    y_pred = np.random.choice([0, 1], size=test_settings.N_SAMPLES)
    return y_true, y_pred


@pytest.fixture
def sample_probabilities(test_settings: TestSettings) -> np.ndarray:
    """Provides sample probability predictions for evaluation testing.

    Args:
        test_settings (TestSettings): Test settings fixture.
    
    Returns:
        np.ndarray: Probability predictions with shape (n_samples, n_classes).
    """
    np.random.seed(test_settings.SEED)
    probs = np.random.rand(test_settings.N_SAMPLES, 2)
    # Normalize to make valid probabilities
    return probs / probs.sum(axis=1, keepdims=True)
