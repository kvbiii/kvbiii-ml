# ruff: noqa: ANN401
from typing import Any, Protocol


class Estimator(Protocol):
    """
    Model the trainers can drive: sklearn, CatBoost, LightGBM and XGBoost classifiers.

    Every method takes ``*args: Any, **kwargs: Any`` on purpose: the libraries share these method
    names but not their signatures, and only this form accepts all of them. Regressors have no
    ``predict_proba``; generic code calls it only on classification problems.
    """

    def fit(self, *args: Any, **kwargs: Any) -> Any: ...

    def get_params(self, *args: Any, **kwargs: Any) -> Any: ...

    def predict(self, *args: Any, **kwargs: Any) -> Any: ...

    def predict_proba(self, *args: Any, **kwargs: Any) -> Any: ...

    def set_params(self, *args: Any, **kwargs: Any) -> Any: ...
