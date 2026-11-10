"""Tests for kvbiii_ml._validation module."""

import pytest
from sklearn.exceptions import NotFittedError

from kvbiii_ml._validation import check_fitted


class _Tuner:
    """Stand-in whose result attribute appears only after tune()."""

    def tune(self) -> None:
        """Set the result attribute."""
        self.best_ = 1


def test_check_fitted_raises_until_method_sets_attribute():
    """Tests check_fitted raises NotFittedError naming the method, then passes.

    Asserts:
        - NotFittedError names the class and the method before tune()
        - No error once every attribute exists
    """
    tuner = _Tuner()
    with pytest.raises(
        NotFittedError, match=r"This _Tuner instance is not fitted yet\. Call tune\(\)"
    ):
        check_fitted(tuner, "best_", method="tune")
    tuner.tune()
    check_fitted(tuner, "best_", method="tune")
