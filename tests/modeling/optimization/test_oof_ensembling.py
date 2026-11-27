"""Tests for kvbiii_ml.modeling.optimization.oof_ensembling."""

import numpy as np
import pytest
from sklearn.metrics import roc_auc_score

from kvbiii_ml.modeling.optimization.oof_ensembling import (
    BlendResult,
    _auc,
    hill_climb,
    logistic_stack,
)


def _members(n: int = 3000, seed: int = 0) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return (oof, y, folds, test): 4 members of decreasing signal, 5 fold ids."""
    rng = np.random.default_rng(seed)
    y = rng.integers(0, 2, n)
    signal = np.array([1.5, 1.0, 0.6, 0.0])
    oof = y[:, None] * signal + rng.normal(size=(n, 4))
    test = rng.normal(size=(n // 3, 4))
    folds = rng.permutation(np.arange(n) % 5)
    return oof, y, folds, test


@pytest.mark.parametrize("space", ["rank", "logit"])
def test_hill_climb_gives_all_weight_to_a_perfect_member(space):
    """Tests the known answer: a member that separates the classes perfectly takes all the weight.

    Asserts:
        - weights are one-hot on the perfect member, nested score is 1
    """
    rng = np.random.default_rng(1)
    y = rng.integers(0, 2, 2000)
    perfect = np.clip(0.2 + 0.6 * y + rng.uniform(-0.1, 0.1, 2000), 0, 1)
    oof = np.column_stack([rng.uniform(size=2000), perfect, rng.uniform(size=2000)])
    folds = np.arange(2000) % 5

    result = hill_climb(oof, y, folds, oof[:100], space=space)

    if not np.array_equal(result.weights, [0.0, 1.0, 0.0]) or result.score != 1.0:
        raise AssertionError(result)


def test_logistic_stack_puts_the_largest_coefficient_on_the_perfect_member():
    """Tests the known answer for the stack.

    Asserts:
        - The perfect member has the largest coefficient and the nested score is ~1
    """
    rng = np.random.default_rng(1)
    y = rng.integers(0, 2, 2000)
    perfect = np.clip(0.2 + 0.6 * y + rng.uniform(-0.1, 0.1, 2000), 0, 1)
    oof = np.column_stack([rng.uniform(size=2000), perfect, rng.uniform(size=2000)])
    folds = np.arange(2000) % 5

    result = logistic_stack(oof, y, folds, oof[:100])

    if int(np.argmax(np.abs(result.weights))) != 1 or result.score < 0.999:
        raise AssertionError(result)


@pytest.mark.parametrize("blend", [hill_climb, logistic_stack])
def test_blend_scores_each_fold_with_a_blend_fitted_without_it(blend):
    """Tests the nested (fold-honest) score.

    Asserts:
        - fold_scores[k] is the AUC of the nested OOF on fold k and score is their mean
        - Shuffling fold 0's labels leaves fold 0's nested OOF unchanged (fitted without it)
        - Output shapes: oof (n,), test (n_test,), weights (m,)
    """
    oof, y, folds, test = _members()
    result = blend(oof, y, folds, test)

    for k in range(5):
        mask = folds == k
        if not np.isclose(result.fold_scores[k], roc_auc_score(y[mask], result.oof[mask])):
            raise AssertionError(k)
    if not np.isclose(result.score, result.fold_scores.mean()):
        raise AssertionError(result.score)
    if result.oof.shape != y.shape or result.test.shape != (len(test),):
        raise AssertionError(result)
    if result.weights.shape != (4,):
        raise AssertionError(result.weights)

    y_shuffled = y.copy()
    fold0 = np.flatnonzero(folds == 0)
    y_shuffled[fold0] = np.random.default_rng(5).permutation(y[fold0])
    shuffled = blend(oof, y_shuffled, folds, test)
    if not np.allclose(shuffled.oof[fold0], result.oof[fold0]):
        raise AssertionError("fold 0's blend saw fold 0's labels")


def test_hill_climb_prefers_the_strongest_member_and_beats_it_with_others():
    """Tests the climb starts at the best member and only adds members that help.

    Asserts:
        - The strongest member has the largest weight, the pure-noise member gets none
        - Weights are non-negative and sum to 1
    """
    oof, y, folds, test = _members(n=20000)
    result = hill_climb(oof, y, folds, test)

    if int(np.argmax(result.weights)) != 0 or result.weights[3] != 0:
        raise AssertionError(result.weights)
    if (result.weights < 0).any() or not np.isclose(result.weights.sum(), 1):
        raise AssertionError(result.weights)


def test_auc_matches_sklearn_with_ties():
    """Tests the fast tie-aware AUC equals roc_auc_score.

    Asserts:
        - Equal to sklearn on scores rounded to create many ties
    """
    rng = np.random.default_rng(2)
    y = rng.integers(0, 2, 5000)
    score = np.round(y * 0.3 + rng.normal(size=5000), 1)

    if not np.isclose(_auc(score, y.astype(bool)), roc_auc_score(y, score)):
        raise AssertionError()


def test_blend_rejects_mismatched_shapes():
    """Tests inputs are validated.

    Asserts:
        - ValueError when y, folds or test do not match oof
    """
    oof, y, folds, test = _members(n=100)
    with pytest.raises(ValueError, match="rows"):
        hill_climb(oof, y[:-1], folds, test)
    with pytest.raises(ValueError, match="members"):
        logistic_stack(oof, y, folds, test[:, :2])


def test_blend_result_is_frozen():
    """Tests BlendResult is immutable.

    Asserts:
        - Assigning a field raises
    """
    oof, y, folds, test = _members(n=500)
    result: BlendResult = hill_climb(oof, y, folds, test)
    with pytest.raises(AttributeError):
        result.score = 0.0  # ty: ignore[invalid-assignment]
