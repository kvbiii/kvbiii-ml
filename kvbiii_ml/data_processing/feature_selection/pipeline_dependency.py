from copy import deepcopy

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline


class _ProcessedColumnSelector(BaseEstimator, TransformerMixin):
    """Select a subset of processed columns from pipeline output."""

    def __init__(self, columns: list[str], all_processed_columns: list[str]) -> None:
        """Initialize selector with the target and full processed column names."""
        self.columns = columns
        self.all_processed_columns = all_processed_columns

    def fit(self, X: pd.DataFrame | np.ndarray, y: pd.Series | None = None) -> "_ProcessedColumnSelector":
        """No-op fit, implemented for sklearn compatibility."""
        self.columns_ = list(self.columns)
        return self

    def transform(self, X: pd.DataFrame | np.ndarray) -> pd.DataFrame:
        """Return only selected processed columns."""
        if isinstance(X, pd.DataFrame):
            return X.loc[:, self.columns_]
        x_df = pd.DataFrame(X, columns=self.all_processed_columns)
        return x_df.loc[:, self.columns_]


class PipelineDependencyGraph:
    """Holds processed feature metadata and builds step-restricted pipelines."""

    def __init__(
        self,
        pipeline: Pipeline | None,
        raw_columns: list[str],
        processed_columns: list[str],
        processed_dtypes: pd.Series,
    ) -> None:
        """Initialize dependency graph container."""
        self.pipeline = pipeline
        self.raw_columns = raw_columns
        self.processed_columns = processed_columns
        self.processed_dtypes = processed_dtypes

    @staticmethod
    def _to_dataframe(
        transformed: pd.DataFrame | np.ndarray,
        feature_names: np.ndarray | list[str] | None,
    ) -> pd.DataFrame:
        """Convert pipeline output to a DataFrame with stable column names."""
        if isinstance(transformed, pd.DataFrame):
            return transformed.copy()
        if feature_names is not None:
            return pd.DataFrame(transformed, columns=list(feature_names))
        n_features = transformed.shape[1] if transformed.ndim == 2 else 0
        generated_columns = [f"feature_{idx}" for idx in range(n_features)]
        return pd.DataFrame(transformed, columns=generated_columns)

    @classmethod
    def build(
        cls,
        pipeline: Pipeline | None,
        X: pd.DataFrame,
        y: pd.Series,
    ) -> "PipelineDependencyGraph":
        """Fit a baseline pipeline clone and capture processed feature metadata."""
        raw_columns = list(X.columns)
        if pipeline is None:
            processed_df = X.copy()
            return cls(
                pipeline=None,
                raw_columns=raw_columns,
                processed_columns=list(processed_df.columns),
                processed_dtypes=processed_df.dtypes,
            )

        pipeline_clone = deepcopy(pipeline)
        transformed = pipeline_clone.fit_transform(X.copy(), y)
        feature_names = None
        if not isinstance(transformed, pd.DataFrame) and hasattr(
            pipeline_clone, "get_feature_names_out"
        ):
            try:
                feature_names = pipeline_clone.get_feature_names_out()
            except AttributeError:
                feature_names = None
        processed_df = cls._to_dataframe(transformed, feature_names)
        return cls(
            pipeline=deepcopy(pipeline),
            raw_columns=raw_columns,
            processed_columns=list(processed_df.columns),
            processed_dtypes=processed_df.dtypes,
        )

    def restrict(self, current_processed_features: list[str]) -> tuple[list[str], Pipeline | None]:
        """Return raw feature set and a pipeline restricted to current processed columns."""
        if self.pipeline is None:
            return list(current_processed_features), None

        restricted_pipeline = deepcopy(self.pipeline)
        restricted_pipeline.steps.append(
            (
                "processed_column_selector",
                _ProcessedColumnSelector(
                    columns=list(current_processed_features),
                    all_processed_columns=list(self.processed_columns),
                ),
            )
        )
        return list(self.raw_columns), restricted_pipeline
