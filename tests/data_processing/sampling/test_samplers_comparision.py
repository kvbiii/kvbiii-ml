"""Tests for FoldwiseSampler class in samplers_comparision module."""

from unittest.mock import Mock, patch

import numpy as np
import pandas as pd
import pytest

from kvbiii_ml.data_processing.sampling.samplers_comparision import FoldwiseSampler


class TestFoldwiseSampler:
    """Test suite for FoldwiseSampler class."""

    def test_foldwisesampler_init_default_parameters(self):
        """Tests FoldwiseSampler initialization with default parameters.

        Asserts:
            - Default strategy is 'none'
            - Default sampler_params is empty dict
            - Sampler is None initially
        """
        sampler = FoldwiseSampler()

        if not (sampler.strategy == "none"):
            raise AssertionError()
        if not (sampler.sampler_params == {}):
            raise AssertionError()
        if not (sampler.sampler is None):
            raise AssertionError()

    def test_foldwisesampler_init_custom_parameters(self):
        """Tests FoldwiseSampler initialization with custom parameters.

        Asserts:
            - Custom strategy is set correctly
            - Custom sampler_params are preserved
        """
        params = {"random_state": 42, "sampling_strategy": 0.5}
        sampler = FoldwiseSampler(strategy="random_over", sampler_params=params)

        if not (sampler.strategy == "random_over"):
            raise AssertionError()
        if not (sampler.sampler_params == params):
            raise AssertionError()

    def test_foldwisesampler_init_none_sampler_params(self):
        """Tests FoldwiseSampler initialization with None sampler_params.

        Asserts:
            - None sampler_params are converted to empty dict
        """
        sampler = FoldwiseSampler(strategy="smote", sampler_params=None)

        if not (sampler.sampler_params == {}):
            raise AssertionError()

    def test_foldwisesampler_fit_none_strategy(self, binary_classification_data):
        """Tests fit method with 'none' strategy.

        Args:
            binary_classification_data (tuple): Binary classification dataset fixture

        Asserts:
            - Sampler remains None for 'none' strategy
        """
        X, y = binary_classification_data
        sampler = FoldwiseSampler(strategy="none")

        sampler.fit(X, y)

        if not (sampler.sampler is None):
            raise AssertionError()

    @patch("kvbiii_ml.data_processing.sampling.samplers_comparision.SMOTENC")
    def test_foldwisesampler_fit_smote_strategy(
        self, mock_smotenc, binary_classification_data
    ):
        """Tests fit method with 'smote' strategy.

        Args:
            mock_smotenc: Mock SMOTENC class
            binary_classification_data (tuple): Binary classification dataset fixture

        Asserts:
            - SMOTENC sampler is created with correct parameters
        """
        X, y = binary_classification_data
        params = {"random_state": 42}
        sampler = FoldwiseSampler(strategy="smote", sampler_params=params)

        sampler.fit(X, y)

        mock_smotenc.assert_called_once_with(**params)
        if not (sampler.sampler == mock_smotenc.return_value):
            raise AssertionError()

    @patch("kvbiii_ml.data_processing.sampling.samplers_comparision.RandomOverSampler")
    def test_foldwisesampler_fit_random_over_strategy(
        self, mock_ros, binary_classification_data
    ):
        """Tests fit method with 'random_over' strategy.

        Args:
            mock_ros: Mock RandomOverSampler class
            binary_classification_data (tuple): Binary classification dataset fixture

        Asserts:
            - RandomOverSampler is created with correct parameters
        """
        X, y = binary_classification_data
        params = {"random_state": 17}
        sampler = FoldwiseSampler(strategy="random_over", sampler_params=params)

        sampler.fit(X, y)

        mock_ros.assert_called_once_with(**params)
        if not (sampler.sampler == mock_ros.return_value):
            raise AssertionError()

    @patch("kvbiii_ml.data_processing.sampling.samplers_comparision.RandomUnderSampler")
    def test_foldwisesampler_fit_random_under_strategy(
        self, mock_rus, binary_classification_data
    ):
        """Tests fit method with 'random_under' strategy.

        Args:
            mock_rus: Mock RandomUnderSampler class
            binary_classification_data (tuple): Binary classification dataset fixture

        Asserts:
            - RandomUnderSampler is created with correct parameters
        """
        X, y = binary_classification_data
        params = {"random_state": 123}
        sampler = FoldwiseSampler(strategy="random_under", sampler_params=params)

        sampler.fit(X, y)

        mock_rus.assert_called_once_with(**params)
        if not (sampler.sampler == mock_rus.return_value):
            raise AssertionError()

    def test_foldwisesampler_fit_unknown_strategy_raises_error(
        self, binary_classification_data
    ):
        """Tests fit method with unknown strategy raises ValueError.

        Args:
            binary_classification_data (tuple): Binary classification dataset fixture

        Asserts:
            - ValueError is raised for unknown strategy
        """
        X, y = binary_classification_data
        sampler = FoldwiseSampler(strategy="unknown_strategy")

        with pytest.raises(
            ValueError, match="Unknown sampling strategy: unknown_strategy"
        ):
            sampler.fit(X, y)

    def test_foldwisesampler_fit_resample_none_strategy(
        self, binary_classification_data
    ):
        """Tests fit_resample method with 'none' strategy returns original data.

        Args:
            binary_classification_data (tuple): Binary classification dataset fixture

        Asserts:
            - Original data is returned unchanged for 'none' strategy
            - X and y maintain same shape and content
        """
        X, y = binary_classification_data
        sampler = FoldwiseSampler(strategy="none")

        resampled_x, resampled_y = sampler.fit_resample(X, y)

        pd.testing.assert_frame_equal(resampled_x, X)
        pd.testing.assert_series_equal(resampled_y, y)

    def test_foldwisesampler_fit_resample_with_mock_sampler(
        self, binary_classification_data
    ):
        """Tests fit_resample method with mock sampler.

        Args:
            binary_classification_data (tuple): Binary classification dataset fixture

        Asserts:
            - Sampler's fit_resample method is called
            - Results are converted back to DataFrame and Series
            - Column names and index are preserved
        """
        X, y = binary_classification_data

        # Create mock sampler
        mock_sampler = Mock()
        # Match the number of features in the binary_classification_data fixture
        resampled_x = np.random.rand(3, X.shape[1])  # Same number of features
        resampled_y = np.array([0, 1, 0])
        mock_sampler.fit_resample.return_value = (resampled_x, resampled_y)

        sampler = FoldwiseSampler(strategy="none")
        sampler.sampler = mock_sampler

        result_x, result_y = sampler.fit_resample(X, y)

        mock_sampler.fit_resample.assert_called_once_with(X, y)
        if not (isinstance(result_x, pd.DataFrame)):
            raise AssertionError()
        if not (isinstance(result_y, pd.Series)):
            raise AssertionError()
        if not (list(result_x.columns) == list(X.columns)):
            raise AssertionError()
        if not (result_y.name == y.name):
            raise AssertionError()

    def test_foldwisesampler_fit_transform_calls_fit_and_fit_resample(
        self, binary_classification_data
    ):
        """Tests fit_transform method calls both fit and fit_resample.

        Args:
            binary_classification_data (tuple): Binary classification dataset fixture

        Asserts:
            - fit_transform equivalent to calling fit then fit_resample
            - Returns same result as separate method calls
        """
        X, y = binary_classification_data

        sampler1 = FoldwiseSampler(strategy="none")
        result1 = sampler1.fit_transform(X, y)

        sampler2 = FoldwiseSampler(strategy="none")
        sampler2.fit(X, y)
        result2 = sampler2.fit_resample(X, y)

        pd.testing.assert_frame_equal(result1[0], result2[0])
        pd.testing.assert_series_equal(result1[1], result2[1])

    @patch("kvbiii_ml.data_processing.sampling.samplers_comparision.RandomOverSampler")
    def test_foldwisesampler_integration_random_over(
        self, mock_ros, binary_classification_data
    ):
        """Tests complete workflow with RandomOverSampler strategy.

        Args:
            mock_ros: Mock RandomOverSampler class
            binary_classification_data (tuple): Binary classification dataset fixture

        Asserts:
            - Complete workflow from initialization to resampling works
            - Mock sampler is properly configured and called
        """
        X, y = binary_classification_data

        # Configure mock
        mock_instance = Mock()
        mock_ros.return_value = mock_instance
        mock_instance.fit_resample.return_value = (X.values, y.values)

        sampler = FoldwiseSampler(
            strategy="random_over", sampler_params={"random_state": 42}
        )
        sampler.fit_transform(X, y)

        mock_ros.assert_called_once_with(random_state=42)
        mock_instance.fit_resample.assert_called_once_with(X, y)

    def test_foldwisesampler_preserves_dataframe_structure(
        self, binary_classification_data
    ):
        """Tests that DataFrame structure is preserved after resampling.

        Args:
            binary_classification_data (tuple): Binary classification dataset fixture

        Asserts:
            - Column names are preserved
            - Series name is preserved
            - Data types are maintained
        """
        X, y = binary_classification_data
        original_columns = X.columns.tolist()
        original_name = y.name

        sampler = FoldwiseSampler(strategy="none")
        result_x, result_y = sampler.fit_resample(X, y)

        if not (result_x.columns.tolist() == original_columns):
            raise AssertionError()
        if not (result_y.name == original_name):
            raise AssertionError()
        if not (all(result_x.dtypes == X.dtypes)):
            raise AssertionError()

    def test_foldwisesampler_empty_sampler_params_handling(
        self, binary_classification_data
    ):
        """Tests handling of empty sampler parameters.

        Args:
            binary_classification_data (tuple): Binary classification dataset fixture

        Asserts:
            - Empty parameters are handled correctly
            - No errors occur with empty parameter dict
        """
        X, y = binary_classification_data

        with patch(
            "kvbiii_ml.data_processing.sampling.samplers_comparision.RandomOverSampler"
        ) as mock_ros:
            sampler = FoldwiseSampler(strategy="random_over", sampler_params={})
            sampler.fit(X, y)

            mock_ros.assert_called_once_with()

    def test_foldwisesampler_strategy_case_sensitivity(
        self, binary_classification_data
    ):
        """Tests that strategy parameter is case-sensitive.

        Args:
            binary_classification_data (tuple): Binary classification dataset fixture

        Asserts:
            - Case-sensitive strategy names raise errors for incorrect case
        """
        X, y = binary_classification_data

        sampler = FoldwiseSampler(strategy="NONE")  # Wrong case

        with pytest.raises(ValueError, match="Unknown sampling strategy: NONE"):
            sampler.fit(X, y)


if __name__ == "__main__":
    print("Run this file with pytest to execute tests.")
