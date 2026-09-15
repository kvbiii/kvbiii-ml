import warnings
from dataclasses import dataclass

import pandas as pd
from sklearn.base import BaseEstimator, clone
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer

_RAW_SELECTOR_STEP = "_raw_selector"
_FEATURE_SELECTOR_STEP = "_feature_selector"


def _select_columns(X: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    """Selects specified columns from X, silently skipping any absent ones.

    Args:
        X (pd.DataFrame): Input features.
        features (list[str]): Column names to select.

    Returns:
        pd.DataFrame: X restricted to the requested columns.
    """
    return X[[f for f in features if f in X.columns]]


def _is_internal_selector_step(name: str, step: BaseEstimator) -> bool:
    """Returns whether a step is restrict()'s own bookkeeping selector, not a user step.

    Args:
        name (str): The step's name within its pipeline.
        step (BaseEstimator): The step itself.

    Returns:
        bool: True if this step was added by a prior restrict() call.
    """
    return (
        name in (_RAW_SELECTOR_STEP, _FEATURE_SELECTOR_STEP)
        and isinstance(step, FunctionTransformer)
        and step.func is _select_columns
    )


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
    step_derivations: list[tuple[str, dict[str, list[str]]]]

    @classmethod
    def build(
        cls,
        pipeline: Pipeline | None,
        X: pd.DataFrame,
        y: pd.Series | None = None,
    ) -> "PipelineDependencyGraph":
        """Fits the full pipeline once on real data and extracts its dependency graph.

        Walks the pipeline exactly as given, including any
        _raw_selector/_feature_selector bookkeeping steps a prior restrict()
        call already added - so processed_columns always reflects what the
        pipeline truly outputs right now, even when it was already restricted
        once. restrict() (not build()) is responsible for not duplicating
        those bookkeeping steps when it adds its own.

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
            return cls(raw_columns, list(raw_columns), None, identity, None, [])

        node_to_raw: dict[str, frozenset[str]] = {
            column: frozenset({column}) for column in raw_columns
        }
        step_derivations: list[tuple[str, dict[str, list[str]]]] = []
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
                step_derivations.append((step_name, new_sources))

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
            step_derivations,
        )

    def restrict(
        self, processed_features: list[str]
    ) -> tuple[list[str], Pipeline | None]:
        """Prunes the graph to the minimal raw inputs and pipeline needed.

        Steps with an explicit ``variables`` list are filtered to only the
        source columns whose OWN derived output from that specific step is
        still needed - not every raw column needed anywhere in the pipeline,
        since a column can be a required pass-through or another step's input
        without that step's own transformation of it ever being used. Steps
        are dropped entirely if their filtered list becomes empty. Steps with
        ``variables=None`` (auto-detect) are left unmodified - they naturally
        auto-detect fewer columns once fed fewer raw inputs, so no trial-fit
        is ever needed to "check compatibility". Any _raw_selector/
        _feature_selector bookkeeping steps a prior restrict() call already
        added are dropped and replaced by this call's own, rather than kept
        alongside them - otherwise re-restricting an already-restricted
        pipeline would collide on duplicate step names.

        Args:
            processed_features (list[str]): Processed columns the caller wants
                the restricted pipeline to still produce.

        Returns:
            tuple[list[str], Pipeline | None]: Raw columns needed, in the
                graph's original column order, and a cloned restricted
                Pipeline starting and ending in a column-selector step, or
                None when this graph has no pipeline.
        """
        raw_needed: set[str] = set()
        for column in processed_features:
            raw_needed |= self.processed_to_raw.get(column, {column})
        raw_needed_list = [
            column for column in self.raw_columns if column in raw_needed
        ]

        if self.pipeline is None:
            return raw_needed_list, None

        step_survivors = self._resolve_step_survivors(processed_features)

        new_steps: list[tuple[str, BaseEstimator]] = [
            (
                _RAW_SELECTOR_STEP,
                FunctionTransformer(
                    func=_select_columns, kw_args={"features": raw_needed_list}
                ),
            )
        ]
        for name, step in self.pipeline.steps:
            if _is_internal_selector_step(name, step):
                continue
            cloned_step = clone(step)
            variables = cloned_step.get_params().get("variables")
            if isinstance(variables, list):
                survivors = step_survivors.get(name, set())
                filtered = [variable for variable in variables if variable in survivors]
                if not filtered:
                    continue
                cloned_step.set_params(variables=filtered)
            new_steps.append((name, cloned_step))

        new_steps.append(
            (
                _FEATURE_SELECTOR_STEP,
                FunctionTransformer(
                    func=_select_columns,
                    kw_args={"features": list(processed_features)},
                ),
            )
        )
        return raw_needed_list, Pipeline(new_steps)

    def _resolve_step_survivors(
        self, processed_features: list[str]
    ) -> dict[str, set[str]]:
        """Walks step_derivations backward to find each step's still-needed source columns.

        Args:
            processed_features (list[str]): Processed columns the caller wants
                the restricted pipeline to still produce.

        Returns:
            dict[str, set[str]]: Step name mapped to the source columns whose
                own derived output from that step is still required.
        """
        needed: set[str] = set(processed_features)
        step_survivors: dict[str, set[str]] = {}
        for step_name, new_sources in reversed(self.step_derivations):
            next_needed: set[str] = set()
            survivors: set[str] = set()
            for column in needed:
                sources = new_sources.get(column)
                if sources is None:
                    next_needed.add(column)
                    continue
                survivors.update(sources)
                next_needed.update(sources)
            step_survivors[step_name] = survivors
            needed = next_needed
        return step_survivors


def build_restricted_pipeline(
    pipeline: Pipeline | None,
    X: pd.DataFrame,
    selected_features: list[str],
    y: pd.Series | None = None,
) -> Pipeline:
    """Builds a dependency graph and returns a pipeline restricted to selected_features.

    Convenience wrapper around PipelineDependencyGraph.build() followed by
    restrict() for callers that only need the restricted pipeline itself and
    have no other use for the intermediate graph. Always returns a real,
    fit_transform-ready Pipeline, including when pipeline is None - unlike
    restrict() itself, which returns None in that case.

    Args:
        pipeline (Pipeline | None): Preprocessing pipeline template, or None
            to build only a column-selecting pipeline.
        X (pd.DataFrame): Full raw feature matrix.
        selected_features (list[str]): Processed columns the returned pipeline
            should produce.
        y (pd.Series | None, optional): Target forwarded to each step's fit()
            during the graph build. Defaults to None.

    Returns:
        Pipeline: A restricted pipeline ready to fit_transform() directly on
            the full raw X, ending in a column selector limited to
            selected_features.
    """
    graph = PipelineDependencyGraph.build(pipeline, X, y)
    _, restricted = graph.restrict(list(selected_features))
    if restricted is None:
        restricted = Pipeline(
            [
                (
                    _FEATURE_SELECTOR_STEP,
                    FunctionTransformer(
                        func=_select_columns,
                        kw_args={"features": list(selected_features)},
                    ),
                )
            ]
        )
    return restricted
