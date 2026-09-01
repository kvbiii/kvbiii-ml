import numpy as np
import pandas as pd
from feature_engine.encoding import MeanEncoder
from feature_engine.imputation import MeanMedianImputer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline

from kvbiii_ml.data_processing.feature_engineering.categorical_aligner import (
    CategoricalAligner,
)
from kvbiii_ml.data_processing.feature_selection.pipeline_dependency import (
    PipelineDependencyGraph,
    _resolve_new_column_sources,
)
from kvbiii_ml.data_processing.preprocessing.categorical_encoding.string_similarity_encoder import (
    StringSimilarityEncoderWithOriginal,
)
from kvbiii_ml.data_processing.preprocessing.discretisation.equal_width_discretiser import (
    EqualWidthDiscretiserWithOriginal,
)
from kvbiii_ml.data_processing.preprocessing.outlier_handling.winsorizer_trimmer import (
    WinsorizerWithOriginal,
)


class _NoContractCombiner(BaseEstimator, TransformerMixin):
    """Transformer with no dependency contract that combines two columns into one."""

    def fit(self, X: pd.DataFrame, y: pd.Series | None = None) -> "_NoContractCombiner":
        """Fits the combiner (no-op, unsupervised).

        Args:
            X (pd.DataFrame): Training features.
            y (pd.Series | None, optional): Unused. Defaults to None.

        Returns:
            _NoContractCombiner: Self.
        """
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Appends a new column combining the first two input columns.

        Args:
            X (pd.DataFrame): Features to transform.

        Returns:
            pd.DataFrame: Original columns plus a new "combined" column.
        """
        result = X.copy()
        result["combined"] = X.iloc[:, 0].astype(str) + "_" + X.iloc[:, 1].astype(str)
        return result


class _FakeStepWithContract:
    """Bare object exposing only the get_derived_column_dependencies() contract."""

    def get_derived_column_dependencies(self) -> dict[str, list[str]]:
        """Declares a fixed dependency for a single derived column.

        Returns:
            dict[str, list[str]]: "new_col" mapped to its declared source "a".
        """
        return {"new_col": ["a"]}


def test_build_with_none_pipeline_returns_identity_graph():
    """Tests build(None, X) returns an identity graph with no dtypes and no pipeline.

    Asserts:
        - processed_columns equals raw_columns
        - processed_dtypes and pipeline are both None
        - restrict() returns the requested columns and a None pipeline
    """
    X = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
    graph = PipelineDependencyGraph.build(None, X)

    if graph.processed_columns != list(X.columns):
        raise AssertionError(graph.processed_columns)
    if graph.processed_dtypes is not None:
        raise AssertionError("expected no dtypes for a None pipeline")
    if graph.pipeline is not None:
        raise AssertionError("expected no pipeline stored")

    raw_needed, restricted_pipeline = graph.restrict(["b"])
    if raw_needed != ["b"]:
        raise AssertionError(raw_needed)
    if restricted_pipeline is not None:
        raise AssertionError("expected no restricted pipeline for a None graph")


def test_build_attributes_withoriginal_derived_column_to_its_single_source():
    """Tests a _WithOriginalBase derived column is attributed to its one source raw column.

    Asserts:
        - The derived column maps to exactly its source column
        - An untouched sibling column maps to itself
    """
    X = pd.DataFrame(
        {
            "f0": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            "f1": [10.0, 20.0, 30.0, 40.0, 50.0, 60.0],
        }
    )
    pipeline = Pipeline(
        [("eq_width", EqualWidthDiscretiserWithOriginal(variables=["f0"], bins=3))]
    )
    graph = PipelineDependencyGraph.build(pipeline, X)

    if graph.processed_to_raw["f0_PREPROCESS_EQ_WIDTH"] != frozenset({"f0"}):
        raise AssertionError(graph.processed_to_raw["f0_PREPROCESS_EQ_WIDTH"])
    if graph.processed_to_raw["f1"] != frozenset({"f1"}):
        raise AssertionError(graph.processed_to_raw["f1"])


def test_build_attributes_string_similarity_derived_columns_to_single_source():
    """Tests every StringSimilarityEncoder derived column is attributed to its one source.

    This is the confirmed real bug's regression coverage at the graph level: today's
    suffix-matching never discovers these columns at all.

    Asserts:
        - At least one derived column is discovered
        - Every derived column maps to exactly {"product"}
    """
    X = pd.DataFrame(
        {
            "product": [
                "apple",
                "orange",
                "apple_juice",
                "orange_juice",
                "apple",
                "grape",
            ]
        }
    )
    pipeline = Pipeline(
        [("sse", StringSimilarityEncoderWithOriginal(variables=["product"]))]
    )
    graph = PipelineDependencyGraph.build(pipeline, X)

    derived_columns = [c for c in graph.processed_columns if c != "product"]
    if not derived_columns:
        raise AssertionError("expected at least one derived column")
    for column in derived_columns:
        if graph.processed_to_raw[column] != frozenset({"product"}):
            raise AssertionError(f"{column} not attributed to 'product'")


def test_build_fallback_treats_inplace_transformer_as_identity():
    """Tests transformers with no contract that transform in place get identity edges.

    Covers both a first-party in-place transformer (CategoricalAligner) and a bare
    third-party one (feature_engine's MeanEncoder used directly, not through a
    WithOriginal wrapper) - neither implements get_derived_column_dependencies().

    Asserts:
        - Every output column maps to itself for both transformers
    """
    X = pd.DataFrame({"cat": ["a", "b", "a", "b"], "num": [1, 2, 3, 4]})
    aligner_pipeline = Pipeline(
        [
            (
                "aligner",
                CategoricalAligner(categorical_features=["cat"], warn_on_unknown=False),
            )
        ]
    )
    aligner_graph = PipelineDependencyGraph.build(aligner_pipeline, X)
    if aligner_graph.processed_to_raw["cat"] != frozenset({"cat"}):
        raise AssertionError(aligner_graph.processed_to_raw["cat"])
    if aligner_graph.processed_to_raw["num"] != frozenset({"num"}):
        raise AssertionError(aligner_graph.processed_to_raw["num"])

    X = pd.DataFrame(
        {
            "cat": ["a", "b", "a", "b", "a", "b"],
            "num": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        }
    )
    y_encode = pd.Series([1, 0, 1, 0, 1, 0])
    encoder_pipeline = Pipeline([("mean_enc", MeanEncoder(variables=["cat"]))])
    encoder_graph = PipelineDependencyGraph.build(encoder_pipeline, X, y_encode)
    if encoder_graph.processed_to_raw["cat"] != frozenset({"cat"}):
        raise AssertionError(encoder_graph.processed_to_raw["cat"])


def test_build_fallback_conservatively_unions_all_inputs_for_unattributed_new_column():
    """Tests a genuinely new column from a no-contract transformer depends on all its inputs.

    Asserts:
        - The new "combined" column maps to the union of every input column
    """
    X = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
    pipeline = Pipeline([("combiner", _NoContractCombiner())])
    graph = PipelineDependencyGraph.build(pipeline, X)

    if graph.processed_to_raw["combined"] != frozenset({"a", "b"}):
        raise AssertionError(graph.processed_to_raw["combined"])


def test_resolve_new_column_sources_prefers_contract_over_fallback():
    """Tests the contract's declared dependency is used verbatim when present.

    Asserts:
        - The declared mapping for the new column is returned exactly
        - Unchanged columns are not included in the result
    """
    step = _FakeStepWithContract()
    result = _resolve_new_column_sources(
        step, cols_before=["a", "b"], cols_after=["a", "b", "new_col"]
    )
    if result != {"new_col": ["a"]}:
        raise AssertionError(result)


def test_resolve_new_column_sources_falls_back_to_all_inputs_without_contract():
    """Tests the fallback rule when a step has no dependency contract at all.

    Asserts:
        - The new column is attributed to every input column, in input order
    """
    result = _resolve_new_column_sources(
        object(), cols_before=["a", "b"], cols_after=["a", "b", "new_col"]
    )
    if result != {"new_col": ["a", "b"]}:
        raise AssertionError(result)


def test_restrict_drops_step_when_filtered_variables_list_is_empty():
    """Tests a step whose entire configured variable set is no longer needed is dropped.

    Asserts:
        - raw_needed excludes the now-unnecessary raw column
        - The winsorizer step is entirely absent from the restricted pipeline
    """
    X = pd.DataFrame({"f0": [1.0, 2.0, 3.0, 4.0], "f1": [5.0, 6.0, 7.0, 8.0]})
    pipeline = Pipeline(
        [
            (
                "winsor",
                WinsorizerWithOriginal(
                    variables=["f0"], capping_method="iqr", tail="both", fold=1.5
                ),
            )
        ]
    )
    graph = PipelineDependencyGraph.build(pipeline, X)

    raw_needed, restricted = graph.restrict(["f1"])

    if raw_needed != ["f1"]:
        raise AssertionError(raw_needed)
    step_names = [name for name, _ in restricted.steps]
    if "winsor" in step_names:
        raise AssertionError("expected the winsorizer step to be dropped entirely")


def test_restrict_leaves_variables_none_step_unchanged():
    """Tests an auto-detect (variables=None) step is never set_params-mutated by restrict().

    Asserts:
        - The restricted pipeline's imputer step still has variables=None
    """
    X = pd.DataFrame({"f0": [1.0, 2.0, np.nan, 4.0], "f1": [5.0, 6.0, 7.0, np.nan]})
    pipeline = Pipeline([("imputer", MeanMedianImputer(imputation_method="median"))])
    graph = PipelineDependencyGraph.build(pipeline, X)

    _, restricted = graph.restrict(["f1"])

    imputer_step = dict(restricted.steps)["imputer"]
    if imputer_step.get_params()["variables"] is not None:
        raise AssertionError("expected variables=None to be left unmodified")


def test_restrict_never_drops_a_raw_column_still_needed_transitively():
    """Tests restrict() never prunes a raw column a still-requested output depends on.

    Uses a 2-step pipeline (auto-detect imputer + explicit-list winsorizer) and proves
    the correctness property by actually fitting the restricted pipeline: it must still
    be able to produce the requested derived column from just the raw columns restrict()
    returned.

    Asserts:
        - raw_needed contains exactly the one raw column the target output depends on
        - Fitting the restricted pipeline on just those raw columns still produces
          the requested derived column
    """
    X = pd.DataFrame(
        {
            "age": [20.0, 30.0, 40.0, 50.0, 60.0, 70.0],
            "other": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        }
    )
    pipeline = Pipeline(
        [
            ("imputer", MeanMedianImputer(imputation_method="median")),
            (
                "winsor",
                WinsorizerWithOriginal(
                    variables=["age"], capping_method="iqr", tail="both", fold=1.5
                ),
            ),
        ]
    )
    graph = PipelineDependencyGraph.build(pipeline, X)

    raw_needed, restricted = graph.restrict(["age_PREPROCESS_WINSORIZER"])

    if raw_needed != ["age"]:
        raise AssertionError(raw_needed)
    result = restricted.fit_transform(X[raw_needed])
    if "age_PREPROCESS_WINSORIZER" not in result.columns:
        raise AssertionError(
            "restricted pipeline failed to produce the still-needed column"
        )


if __name__ == "__main__":
    print("Run this file with pytest to execute tests.")
