import numpy as np


def assert_valid_binary_proba_matrix(probabilities: np.ndarray, n_samples: int) -> None:
    """Asserts a binary classification probability matrix is well-formed.

    Shared across CrossValidationTrainer/OOFModel tests since both expose a
    predict_proba producing the same (n_samples, 2) contract.

    Args:
        probabilities (np.ndarray): Output of a predict_proba call.
        n_samples (int): Expected number of rows.

    Raises:
        AssertionError: If the shape, row sums, or value range are invalid.
    """
    if probabilities.shape != (n_samples, 2):
        raise AssertionError(f"Unexpected predict_proba shape: {probabilities.shape}")
    if not np.allclose(probabilities.sum(axis=1), 1.0):
        raise AssertionError("Probability rows do not sum to 1.")
    if not (np.all(probabilities >= 0) and np.all(probabilities <= 1)):
        raise AssertionError("Probabilities fall outside [0, 1].")


if __name__ == "__main__":
    print("This module provides shared test support, not tests to run directly.")
