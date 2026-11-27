"""Blend saved out-of-fold predictions without retraining: Caruana hill climb and logistic stack.

Both functions take the same arrays and return the same ``BlendResult``, so they can be compared
on the same folds:

- ``oof``: ``(n, m)`` out-of-fold scores of m members (probabilities or any monotone score),
- ``y``: ``(n,)`` binary target,
- ``folds``: ``(n,)`` fold id of every row (the folds the members were trained on),
- ``test``: ``(n_test, m)`` test scores of the same members, same column order.

Every reported score is nested: the rows of fold k are scored by a blend fitted only on the other
folds. The returned weights and test blend come from one more fit on all rows. Only binary
ROC AUC is supported.
"""

import os
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np
from scipy.special import logit
from scipy.stats import rankdata
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold

_LOGIT_EPS = 1e-7
_INNER_SPLITS = 5

# fit(x, y) -> (predict(rows) -> scores, fit info kept in BlendResult.params)
FitOne = Callable[[np.ndarray, np.ndarray], tuple[Callable[[np.ndarray], np.ndarray], Any]]


@dataclass(frozen=True)
class BlendResult:
    """Output of ``hill_climb`` and ``logistic_stack``.

    Attributes:
        weights (np.ndarray): ``(m,)`` fitted on all rows; hill-climb weights are non-negative and
            sum to 1, stack weights are the logistic-regression coefficients.
        oof (np.ndarray): ``(n,)`` nested blend scores: fold k scored by a blend fitted without it.
        test (np.ndarray): ``(n_test,)`` the all-rows blend applied to the test matrix.
        fold_scores (np.ndarray): Nested ROC AUC per fold, in sorted fold-id order.
        score (float): Mean of ``fold_scores``.
        params (dict): Hill climb: ``best_iter`` and ``n_iter`` of the all-rows fit and the
            weights of each fold fit. Stack: chosen ``C`` of the all-rows fit and of each fold.
    """

    weights: np.ndarray
    oof: np.ndarray
    test: np.ndarray
    fold_scores: np.ndarray
    score: float
    params: dict = field(default_factory=dict)


def _auc(score: np.ndarray, y: np.ndarray) -> float:
    """ROC AUC of a 1-D score against a boolean target, tie-aware (Mann-Whitney U)."""
    order = np.argsort(score)
    sorted_score = score[order]
    starts = np.r_[0, np.flatnonzero(np.diff(sorted_score)) + 1]
    ends = np.r_[starts[1:], len(score)]
    ranks = np.repeat((starts + ends + 1) / 2, ends - starts)
    positives = y[order]
    n_pos = int(positives.sum())
    n_neg = len(score) - n_pos
    return float((ranks[positives].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def _rank01(matrix: np.ndarray) -> np.ndarray:
    """Column-wise average ranks scaled to (0, 1): ``(rank - 0.5) / n``."""
    return (rankdata(matrix, axis=0) - 0.5) / len(matrix)


def _check_inputs(
    oof: np.ndarray, y: np.ndarray, folds: np.ndarray, test: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Validate shapes and return float/bool/int arrays.

    Raises:
        ValueError: On mismatched rows or members, a non-2-D matrix or a non-binary target.
    """
    oof, test = np.asarray(oof, dtype=float), np.asarray(test, dtype=float)
    y, folds = np.asarray(y), np.asarray(folds)
    if oof.ndim != 2 or test.ndim != 2:  # noqa: PLR2004
        raise ValueError("oof and test must be 2-D (rows, members).")
    if not len(oof) == len(y) == len(folds):
        raise ValueError(
            f"oof, y and folds must have the same rows: {len(oof)}, {len(y)}, {len(folds)}."
        )
    if test.shape[1] != oof.shape[1]:
        raise ValueError(f"test has {test.shape[1]} members, oof has {oof.shape[1]}.")
    if not np.isin(y, (0, 1)).all():
        raise ValueError("y must be binary 0/1.")
    return oof, y.astype(bool), folds, test


def _nested(
    features: np.ndarray, y: np.ndarray, folds: np.ndarray, fit_one: FitOne
) -> tuple[np.ndarray, np.ndarray, list]:
    """Score fold k with ``fit_one`` fitted on the other folds; return (oof, fold_scores, fits)."""
    nested_oof = np.empty(len(y))
    scores, fits = [], []
    for k in np.unique(folds):
        held = folds == k
        predict, info = fit_one(features[~held], y[~held])
        nested_oof[held] = predict(features[held])
        scores.append(_auc(nested_oof[held], y[held]))
        fits.append(info)
    return nested_oof, np.asarray(scores), fits


def _climb(
    features: np.ndarray, y: np.ndarray, *, patience: int, max_iter: int
) -> tuple[np.ndarray, int, int]:
    """Caruana selection with replacement; return (weights at the best iteration, best, total)."""
    features = np.asfortranarray(features)
    n_members = features.shape[1]
    with ThreadPoolExecutor(max_workers=min(n_members, os.cpu_count() or 1)) as pool:

        def scores_of(base: np.ndarray) -> np.ndarray:
            return np.fromiter(
                pool.map(lambda j: _auc(base + features[:, j], y), range(n_members)), float
            )

        single = scores_of(np.zeros(len(y)))
        counts = np.zeros(n_members)
        start = int(np.argmax(single))
        counts[start] = 1
        total = features[:, start].copy()
        best_score, best_counts, best_iter, stalled, iteration = (
            single[start],
            counts.copy(),
            0,
            0,
            0,
        )
        while iteration < max_iter and stalled < patience:
            iteration += 1
            candidates = scores_of(total)
            pick = int(np.argmax(candidates))
            total += features[:, pick]
            counts[pick] += 1
            if candidates[pick] > best_score:
                best_score, best_counts, best_iter, stalled = (
                    candidates[pick],
                    counts.copy(),
                    iteration,
                    0,
                )
            else:
                stalled += 1
    return best_counts / best_counts.sum(), best_iter, iteration


def hill_climb(  # noqa: PLR0913
    oof: np.ndarray,
    y: np.ndarray,
    folds: np.ndarray,
    test: np.ndarray,
    *,
    space: Literal["rank", "logit"] = "rank",
    patience: int = 20,
    max_iter: int = 200,
) -> BlendResult:
    """Caruana hill climb with replacement over member scores, scored fold-honestly.

    Starts from the best single member, then repeatedly adds (with replacement) the member that
    most improves ROC AUC of the running blend. Weights are the selection counts at the best
    iteration; the climb stops after ``patience`` iterations without improvement or at
    ``max_iter``. Weights are non-negative and sum to 1.

    Notes:
    - space: "rank" blends column ranks scaled to (0, 1) (oof and test ranked separately);
      "logit" blends ``logit(p)`` and expects probabilities. Defaults to "rank".

    Args:
        oof (np.ndarray): ``(n, m)`` out-of-fold member scores.
        y (np.ndarray): ``(n,)`` binary target.
        folds (np.ndarray): ``(n,)`` fold id per row.
        test (np.ndarray): ``(n_test, m)`` test member scores.
        space (Literal["rank", "logit"]): Blending space. Defaults to "rank".
        patience (int): Stalled iterations before stopping. Defaults to 20.
        max_iter (int): Maximum additions after the start member. Defaults to 200.

    Returns:
        BlendResult: Weights, nested OOF blend and scores, and the test blend.
    """
    oof, y, folds, test = _check_inputs(oof, y, folds, test)
    if space == "rank":
        features, test_features = _rank01(oof), _rank01(test)
    else:
        features = logit(np.clip(oof, _LOGIT_EPS, 1 - _LOGIT_EPS))
        test_features = logit(np.clip(test, _LOGIT_EPS, 1 - _LOGIT_EPS))

    def fit_one(x: np.ndarray, target: np.ndarray) -> tuple:
        weights, _, _ = _climb(x, target, patience=patience, max_iter=max_iter)
        return (lambda rows: rows @ weights), weights

    nested_oof, fold_scores, fold_weights = _nested(features, y, folds, fit_one)
    weights, best_iter, n_iter = _climb(features, y, patience=patience, max_iter=max_iter)
    return BlendResult(
        weights=weights,
        oof=nested_oof,
        test=test_features @ weights,
        fold_scores=fold_scores,
        score=float(fold_scores.mean()),
        params={"best_iter": best_iter, "n_iter": n_iter, "fold_weights": fold_weights},
    )


def _pick_c(
    x: np.ndarray, y: np.ndarray, c_grid: tuple[float, ...], inner_seeds: tuple[int, ...]
) -> float:
    """Return the C with the best mean inner-CV AUC over StratifiedKFold(5) for each seed."""
    tasks = [
        (c, train, valid)
        for seed in inner_seeds
        for train, valid in StratifiedKFold(_INNER_SPLITS, shuffle=True, random_state=seed).split(
            x, y
        )
        for c in c_grid
    ]

    def score(task: tuple) -> float:
        c, train, valid = task
        model = LogisticRegression(C=c, tol=1e-6, max_iter=1000).fit(x[train], y[train])
        return _auc(model.decision_function(x[valid]), y[valid])

    with ThreadPoolExecutor(max_workers=min(len(tasks), os.cpu_count() or 1)) as pool:
        scores = np.fromiter(pool.map(score, tasks), float).reshape(-1, len(c_grid))
    return float(c_grid[int(np.argmax(scores.mean(axis=0)))])


def logistic_stack(  # noqa: PLR0913
    oof: np.ndarray,
    y: np.ndarray,
    folds: np.ndarray,
    test: np.ndarray,
    *,
    C_grid: tuple[float, ...] = (0.003, 0.01, 0.03, 0.05, 0.1),  # noqa: N803
    inner_seeds: tuple[int, ...] = (0, 1, 2),
) -> BlendResult:
    """Logistic regression on logit-ranks, with C chosen by nested CV.

    Features are ``logit((rank - 0.5) / n)`` of each member column (oof and test ranked
    separately). For each outer fold, C is picked from ``C_grid`` by the mean AUC of an inner
    StratifiedKFold(5) on the other folds, repeated for every seed in ``inner_seeds``; the model
    refitted with that C scores the held-out fold. The all-rows fit picks C the same way.

    Args:
        oof (np.ndarray): ``(n, m)`` out-of-fold member scores.
        y (np.ndarray): ``(n,)`` binary target.
        folds (np.ndarray): ``(n,)`` fold id per row.
        test (np.ndarray): ``(n_test, m)`` test member scores.
        C_grid (tuple[float, ...]): Inverse L2 strengths to choose from; shrink C as the pool
            widens. Defaults to (0.003, 0.01, 0.03, 0.05, 0.1).
        inner_seeds (tuple[int, ...]): Shuffle seeds of the inner C-selection splits.

    Returns:
        BlendResult: Coefficients, nested OOF decision scores, and the test decision scores.
    """
    oof, y, folds, test = _check_inputs(oof, y, folds, test)
    features, test_features = logit(_rank01(oof)), logit(_rank01(test))

    def fit_one(x: np.ndarray, target: np.ndarray) -> tuple:
        c = _pick_c(x, target, C_grid, inner_seeds)
        model = LogisticRegression(C=c, tol=1e-6, max_iter=1000).fit(x, target)
        return model.decision_function, (c, model)

    nested_oof, fold_scores, fold_fits = _nested(features, y, folds, fit_one)
    predict, (c_all, model) = fit_one(features, y)
    return BlendResult(
        weights=model.coef_[0].copy(),
        oof=nested_oof,
        test=predict(test_features),
        fold_scores=fold_scores,
        score=float(fold_scores.mean()),
        params={"C": c_all, "fold_C": [c for c, _ in fold_fits]},
    )


__all__ = ["BlendResult", "hill_climb", "logistic_stack"]
