# ruff: noqa: ANN401
from collections.abc import Callable
from typing import Any, Literal, Protocol, TypedDict

import pandas as pd


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


StylerFormat = dict[Any, str | Callable[..., Any] | None]

Direction = Literal["maximize", "minimize"]


class RfeSummary(TypedDict):
    """Result of a recursive feature elimination run."""

    selected_features: list[str]
    selected_features_names: list[str]
    history: pd.DataFrame
