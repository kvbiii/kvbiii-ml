import numpy as np
import pandas as pd
from feature_engine.encoding import MeanEncoder
from feature_engine.imputation import MeanMedianImputer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline

from kvbiii_ml.data_processing.feature_engineering.categorical_aligner import (
    CategoricalAligner,
)
from kvbiii_ml.data_processing.preprocessing.categorical_encoding.count_frequency_encoder import (
    CountFrequencyEncoderWithOriginal,
)
from kvbiii_ml.data_processing.preprocessing.categorical_encoding.rare_label_encoder import (
    RareLabelEncoderWithOriginal,
)
from kvbiii_ml.data_processing.preprocessing.categorical_encoding.string_similarity_encoder import (
    StringSimilarityEncoderWithOriginal,
)
from kvbiii_ml.data_processing.preprocessing.categorical_encoding.target_encoder import (
    TargetEncoderWithOriginal,
)
from kvbiii_ml.data_processing.preprocessing.discretisation.equal_width_discretiser import (
    EqualWidthDiscretiserWithOriginal,
)
from kvbiii_ml.data_processing.preprocessing.outlier_handling.winsorizer_trimmer import (
    WinsorizerWithOriginal,
)
from kvbiii_ml.data_processing.preprocessing.pipeline_dependency import (
    PipelineDependencyGraph,
    _resolve_new_column_sources,
    build_restricted_pipeline,
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

    if graph.processed_to_raw["f0_PREPROCESS_EQ_WIDTH_3"] != frozenset({"f0"}):
        raise AssertionError(graph.processed_to_raw["f0_PREPROCESS_EQ_WIDTH_3"])
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


def test_restrict_narrows_a_steps_variables_to_only_its_own_still_needed_derived_output():
    """Tests restrict() narrows a step's variables to only its own surviving derived output.

    Reproduces the confirmed real-world bug: a step configured with several raw
    columns in `variables` must not keep encoding every one of them just
    because some are needed elsewhere in the pipeline or as final passthrough
    outputs - only the columns whose own derived output from THIS step is
    still requested should survive its `variables` list.

    Asserts:
        - raw_needed still includes all three raw columns (each needed for a
          different reason), proving nothing was wrongly dropped globally
        - The count-frequency step's narrowed variables contains only "cat_a",
          proving cat_b/cat_c were excluded from THIS step specifically
    """
    X = pd.DataFrame(
        {
            "cat_a": ["a1", "a2", "a1", "a2", "a1", "a2"],
            "cat_b": ["b1", "b2", "b1", "b2", "b1", "b2"],
            "cat_c": ["c1", "c2", "c1", "c2", "c1", "c2"],
        }
    )
    pipeline = Pipeline(
        [
            (
                "count_freq",
                CountFrequencyEncoderWithOriginal(
                    variables=["cat_a", "cat_b", "cat_c"]
                ),
            ),
            ("rare_c", RareLabelEncoderWithOriginal(variables=["cat_c"])),
        ]
    )
    graph = PipelineDependencyGraph.build(pipeline, X)

    processed_features = ["cat_a_PREPROCESS_CNT_FREQ", "cat_b", "cat_c_PREPROCESS_RARE"]
    raw_needed, restricted = graph.restrict(processed_features)

    if set(raw_needed) != {"cat_a", "cat_b", "cat_c"}:
        raise AssertionError(raw_needed)
    count_freq_step = dict(restricted.steps)["count_freq"]
    if count_freq_step.get_params()["variables"] != ["cat_a"]:
        raise AssertionError(count_freq_step.get_params()["variables"])


def test_restrict_narrows_a_variables_none_steps_own_survivors():
    """Tests restrict() narrows a variables=None step down to its own survivors too.

    Reproduces the confirmed real-world bug: two encoders share the same raw
    source column - one configured with an explicit `variables` list (whose
    derived output survives selection), the other left on `variables=None`
    auto-detect (whose derived output does NOT survive selection). Because
    the shared column stays legitimately needed as input to the surviving
    encoder, the auto-detect encoder kept auto-detecting and re-encoding it
    too, even though none of ITS OWN output was ever requested - restrict()
    must narrow that auto-detect encoder's own variables down to just its
    surviving derived columns (empty here), not leave `variables=None`
    unmodified just because the step never had an explicit list to begin with.

    Asserts:
        - The auto-detect encoder's own derived column ("shared_PREPROCESS_TARGET_ENC")
          is absent from the restricted pipeline's output
        - The auto-detect encoder step is dropped entirely, since none of its
          own derived output survived selection
        - The surviving explicit-list encoder's derived column is still produced
    """
    X = pd.DataFrame({"shared": ["a", "b"] * 10})
    y = pd.Series([1, 0] * 10)
    pipeline = Pipeline(
        [
            ("count_freq", CountFrequencyEncoderWithOriginal(variables=["shared"])),
            ("target_enc", TargetEncoderWithOriginal()),
        ]
    )
    graph = PipelineDependencyGraph.build(pipeline, X, y)

    processed_features = ["shared_PREPROCESS_CNT_FREQ"]
    raw_needed, restricted = graph.restrict(processed_features)

    if raw_needed != ["shared"]:
        raise AssertionError(raw_needed)
    step_names = [name for name, _ in restricted.steps]
    if "target_enc" in step_names:
        raise AssertionError(
            "expected the auto-detect target encoder step to be dropped entirely"
        )
    result = restricted.fit_transform(X)
    if list(result.columns) != processed_features:
        raise AssertionError(list(result.columns))


def test_restrict_pipeline_is_self_sufficient_on_the_full_raw_x():
    """Tests the restricted pipeline works when fit on the full, unsliced raw X.

    Asserts:
        - fit_transform on the full X (including a column not in raw_needed)
          still produces the requested derived column, proving the leading
          _raw_selector step correctly narrows the input on its own
    """
    X = pd.DataFrame(
        {
            "f0": [1.0, 2.0, 3.0, 4.0],
            "f1": [5.0, 6.0, 7.0, 8.0],
            "unused": [9.0, 10.0, 11.0, 12.0],
        }
    )
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

    _, restricted = graph.restrict(["f0_PREPROCESS_WINSORIZER"])
    result = restricted.fit_transform(X)

    if "f0_PREPROCESS_WINSORIZER" not in result.columns:
        raise AssertionError(
            "restricted pipeline failed to produce the requested column from full X"
        )


def test_build_records_step_derivations_in_forward_order():
    """Tests step_derivations mirrors pipeline.steps order and each step's new columns.

    Asserts:
        - step_derivations has one entry per pipeline step, in the same order
        - Each entry's new-column mapping matches what that step alone declares
    """
    X = pd.DataFrame(
        {
            "f0": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            "f1": [10.0, 20.0, 30.0, 40.0, 50.0, 60.0],
        }
    )
    pipeline = Pipeline(
        [
            ("eq_width", EqualWidthDiscretiserWithOriginal(variables=["f0"], bins=3)),
            (
                "winsor",
                WinsorizerWithOriginal(
                    variables=["f1"], capping_method="iqr", tail="both", fold=1.5
                ),
            ),
        ]
    )
    graph = PipelineDependencyGraph.build(pipeline, X)

    step_names = [name for name, _ in graph.step_derivations]
    if step_names != ["eq_width", "winsor"]:
        raise AssertionError(step_names)
    if graph.step_derivations[0][1] != {"f0_PREPROCESS_EQ_WIDTH_3": ["f0"]}:
        raise AssertionError(graph.step_derivations[0][1])
    if graph.step_derivations[1][1] != {"f1_PREPROCESS_WINSORIZER": ["f1"]}:
        raise AssertionError(graph.step_derivations[1][1])


def test_build_restricted_pipeline_returns_pipeline_when_input_pipeline_is_none():
    """Tests build_restricted_pipeline() always returns a Pipeline, even for pipeline=None.

    Asserts:
        - The return value is a Pipeline instance, never None
        - fit_transform() on it produces exactly the requested columns
    """
    X = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})

    restricted = build_restricted_pipeline(None, X, ["b"])

    if not isinstance(restricted, Pipeline):
        raise AssertionError(type(restricted))
    result = restricted.fit_transform(X)
    if list(result.columns) != ["b"]:
        raise AssertionError(list(result.columns))


def test_build_restricted_pipeline_matches_manual_build_and_restrict():
    """Tests build_restricted_pipeline() matches the manual build() -> restrict() call it wraps.

    Asserts:
        - The convenience function's output columns match the manually built
          restricted pipeline's output columns for the same inputs
    """
    X = pd.DataFrame(
        {
            "cat": ["a", "b", "a", "b", "a", "b"],
            "other": [1, 2, 3, 4, 5, 6],
        }
    )
    pipeline = Pipeline(
        [("count_freq", CountFrequencyEncoderWithOriginal(variables=["cat"]))]
    )
    selected_features = ["cat_PREPROCESS_CNT_FREQ"]

    manual_graph = PipelineDependencyGraph.build(pipeline, X)
    _, manual_restricted = manual_graph.restrict(selected_features)
    manual_result = manual_restricted.fit_transform(X)

    convenience_result = build_restricted_pipeline(
        pipeline, X, selected_features
    ).fit_transform(X)

    if list(convenience_result.columns) != list(manual_result.columns):
        raise AssertionError((convenience_result.columns, manual_result.columns))


def test_build_on_already_restricted_pipeline_reports_true_processed_columns():
    """Tests build() on an already-restricted pipeline never inflates processed_columns.

    Reproduces the confirmed real-world bug: a feature selector (e.g.
    PermutationRecursiveFeatureElimination) calls PipelineDependencyGraph.build()
    on a pipeline that was already produced by build_restricted_pipeline(),
    feeding it the FULL raw X (not pre-sliced to the earlier restrict() call's
    raw_needed). build() must walk the pipeline's own leading _raw_selector and
    trailing _feature_selector exactly as it would really execute - if those
    bookkeeping steps were stripped before walking (an earlier, incorrect fix),
    a raw column the once-restricted pipeline was built to exclude
    ("unused_extra") would incorrectly reappear in processed_columns, inflating
    the reported feature count past what actually reaches the model.

    Asserts:
        - processed_columns exactly matches the once-restricted pipeline's real
          two-column output, not inflated by "unused_extra" or the bare "cat"
          pass-through column the trailing selector already drops
    """
    X = pd.DataFrame(
        {
            "cat": ["a", "b", "a", "b", "a", "b"],
            "num": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            "unused_extra": [7.0, 8.0, 9.0, 10.0, 11.0, 12.0],
        }
    )
    pipeline = Pipeline(
        [("count_freq", CountFrequencyEncoderWithOriginal(variables=["cat"]))]
    )
    once_restricted = build_restricted_pipeline(
        pipeline, X, ["cat_PREPROCESS_CNT_FREQ", "num"]
    )

    second_graph = PipelineDependencyGraph.build(once_restricted, X)

    if second_graph.processed_columns != ["cat_PREPROCESS_CNT_FREQ", "num"]:
        raise AssertionError(second_graph.processed_columns)


def test_build_restricted_pipeline_chains_without_duplicate_step_names():
    """Tests re-restricting an already-restricted pipeline never raises on duplicate names.

    Reproduces the confirmed real-world scenario: build_restricted_pipeline()
    is called once, its result is reassigned to the caller's own pipeline
    variable, then handed to a feature selector that restricts it again
    internally (modelled here by calling build_restricted_pipeline() a second
    time on the already-restricted result with a narrower feature list).

    Asserts:
        - The second call succeeds without a duplicate-step-name ValueError
        - The rebuilt pipeline's step names are all unique
        - fit_transform() on the full raw X still produces exactly the
          narrower, second-call feature list - not a stale first-call one
    """
    X = pd.DataFrame(
        {
            "cat": ["a", "b", "a", "b", "a", "b"],
            "num": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        }
    )
    pipeline = Pipeline(
        [
            ("count_freq", CountFrequencyEncoderWithOriginal(variables=["cat"])),
            (
                "winsor",
                WinsorizerWithOriginal(
                    variables=["num"], capping_method="iqr", tail="both", fold=1.5
                ),
            ),
        ]
    )
    once_restricted = build_restricted_pipeline(
        pipeline, X, ["cat_PREPROCESS_CNT_FREQ", "num"]
    )

    twice_restricted = build_restricted_pipeline(
        once_restricted, X, ["cat_PREPROCESS_CNT_FREQ"]
    )

    step_names = [name for name, _ in twice_restricted.steps]
    if len(step_names) != len(set(step_names)):
        raise AssertionError(step_names)
    result = twice_restricted.fit_transform(X)
    if list(result.columns) != ["cat_PREPROCESS_CNT_FREQ"]:
        raise AssertionError(list(result.columns))


if __name__ == "__main__":
    print("Run this file with pytest to execute tests.")
