import warnings
from dataclasses import dataclass

import pandas as pd
from sklearn.base import BaseEstimator, clone
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer


def _select_columns(X: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    """Selects specified columns from X, silently skipping any absent ones.

    Args:
        X (pd.DataFrame): Input features.
        features (list[str]): Column names to select.

    Returns:
        pd.DataFrame: X restricted to the requested columns.
    """
    return X[[f for f in features if f in X.columns]]


def _resolve_new_column_sources(
    step: BaseEstimator,
    cols_before: list[str],
    cols_after: list[str],
) -> dict[str, list[str]]:
    """Returns, for every new column a step's transform added, its source column(s).

    Prefers the step's own ``get_derived_column_dependencies()`` contract when
    present (duck-typed, no inheritance requirement). Falls back to a safe,
    generic rule for any transformer that doesn't implement it: a genuinely new
    column name is conservatively attributed to every column fed into this
    step, since its true origin can't be known without the explicit contract.
    Pass-through/in-place columns (same name before and after) are handled by
    the caller, not here, since that rule needs no per-step knowledge at all.

    Args:
        step (BaseEstimator): The already-fitted pipeline step (a clone).
        cols_before (list[str]): Column names fed into this step.
        cols_after (list[str]): Column names this step's fit_transform() output.

    Returns:
        dict[str, list[str]]: New output column name mapped to its source
            column name(s). Only contains entries for columns present in
            cols_after but absent from cols_before.
    """
    before_set = set(cols_before)
    new_columns = [column for column in cols_after if column not in before_set]
    if not new_columns:
        return {}

    declared: dict[str, list[str]] = {}
    if hasattr(step, "get_derived_column_dependencies"):
        declared = step.get_derived_column_dependencies()

    return {
        column: declared[column] if column in declared else list(cols_before)
        for column in new_columns
    }


@dataclass
class PipelineDependencyGraph:
    """Maps every post-pipeline processed column to the raw input column(s) it depends on.

    Built from one real fit of the full preprocessing pipeline, letting any
    feature selector ask "given I want to keep these N processed columns,
    what's the minimal restricted pipeline and raw input set?" through a
    single restrict() call, without ever trial-fitting on a throwaway sample.
    """

    raw_columns: list[str]
    processed_columns: list[str]
    processed_dtypes: pd.Series | None
    processed_to_raw: dict[str, frozenset[str]]
    pipeline: Pipeline | None

    @classmethod
    def build(
        cls,
        pipeline: Pipeline | None,
        X: pd.DataFrame,
        y: pd.Series | None = None,
    ) -> "PipelineDependencyGraph":
        """Fits the full pipeline once on real data and extracts its dependency graph.

        Args:
            pipeline (Pipeline | None): Preprocessing pipeline template, or None
                when the caller has no pipeline (identity graph).
            X (pd.DataFrame): Full raw feature matrix.
            y (pd.Series | None, optional): Target forwarded to each step's
                fit(). Defaults to None.

        Returns:
            PipelineDependencyGraph: The built graph.

        Raises:
            TypeError: If any step's fit_transform() does not return a pandas
                DataFrame.
        """
        raw_columns = list(X.columns)
        if pipeline is None:
            identity = {column: frozenset({column}) for column in raw_columns}
            return cls(raw_columns, list(raw_columns), None, identity, None)

        node_to_raw: dict[str, frozenset[str]] = {
            column: frozenset({column}) for column in raw_columns
        }
        x_running = X.copy()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            for step_name, step in pipeline.steps:
                cols_before = list(x_running.columns)
                cloned_step = clone(step)
                x_after = cloned_step.fit_transform(x_running, y)
                if not isinstance(x_after, pd.DataFrame):
                    raise TypeError(
                        f"Pipeline step '{step_name}' ({type(step).__name__}) must "
                        "return a pandas DataFrame from fit_transform() for "
                        f"dependency-graph extraction; got {type(x_after).__name__}."
                    )
                cols_after = list(x_after.columns)
                before_set = set(cols_before)
                new_sources = _resolve_new_column_sources(
                    cloned_step, cols_before, cols_after
                )

                for column in cols_after:
                    if column in before_set:
                        node_to_raw.setdefault(column, frozenset({column}))
                        continue
                    raw_set: set[str] = set()
                    for source in new_sources[column]:
                        raw_set |= node_to_raw.get(source, {source})
                    node_to_raw[column] = frozenset(raw_set)

                x_running = x_after

        processed_columns = list(x_running.columns)
        processed_to_raw = {column: node_to_raw[column] for column in processed_columns}
        return cls(
            raw_columns,
            processed_columns,
            x_running.dtypes.copy(),
            processed_to_raw,
            pipeline,
        )

    def restrict(
        self, processed_features: list[str]
    ) -> tuple[list[str], Pipeline | None]:
        """Prunes the graph to the minimal raw inputs and pipeline needed.

        Steps with an explicit ``variables`` list are filtered to the raw
        columns still needed, and dropped entirely if that filtered list
        becomes empty. Steps with ``variables=None`` (auto-detect) are left
        unmodified - they naturally auto-detect fewer columns once fed fewer
        raw inputs, so no trial-fit is ever needed to "check compatibility".

        Args:
            processed_features (list[str]): Processed columns the caller wants
                the restricted pipeline to still produce.

        Returns:
            tuple[list[str], Pipeline | None]: Raw columns needed, in the
                graph's original column order, and a cloned restricted
                Pipeline ending in a column-selector step, or None when this
                graph has no pipeline.
        """
        raw_needed: set[str] = set()
        for column in processed_features:
            raw_needed |= self.processed_to_raw.get(column, {column})
        raw_needed_list = [
            column for column in self.raw_columns if column in raw_needed
        ]

        if self.pipeline is None:
            return raw_needed_list, None

        new_steps: list[tuple[str, BaseEstimator]] = []
        for name, step in self.pipeline.steps:
            cloned_step = clone(step)
            variables = cloned_step.get_params().get("variables")
            if isinstance(variables, list):
                filtered = [
                    variable for variable in variables if variable in raw_needed
                ]
                if not filtered:
                    continue
                cloned_step.set_params(variables=filtered)
            new_steps.append((name, cloned_step))

        new_steps.append(
            (
                "_feature_selector",
                FunctionTransformer(
                    func=_select_columns,
                    kw_args={"features": list(processed_features)},
                ),
            )
        )
        return raw_needed_list, Pipeline(new_steps)
