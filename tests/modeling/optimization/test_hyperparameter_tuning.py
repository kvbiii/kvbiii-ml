"""Tests for kvbiii_ml.modeling.optimization.hyperparameter_tuning module."""

from unittest.mock import Mock

import numpy as np
import optuna
import pandas as pd
import pytest
from lightgbm import LGBMClassifier, LGBMRegressor
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.datasets import make_classification, make_regression
from sklearn.model_selection import KFold, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline

from kvbiii_ml.modeling.optimization.hyperparameter_tuning import (
    RandomSearchCV,
    _FoldMedianPruner,
)
from kvbiii_ml.modeling.training.cross_validation import CrossValidationTrainer

FIT_TRANSFORM_CALLS: list[str] = []


class CountingPassthrough(BaseEstimator, TransformerMixin):
    """Passthrough transformer that records every fit_transform call globally."""

    def fit(self, X: pd.DataFrame, y: pd.Series | None = None) -> "CountingPassthrough":
        """Marks the transformer fitted so the enclosing Pipeline allows transform.

        Args:
            X (pd.DataFrame): Features, unused beyond marking fitted state.
            y (pd.Series | None): Target, unused. Defaults to None.

        Returns:
            CountingPassthrough: This instance.
        """
        self.fitted_ = True
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Returns the input unchanged.

        Args:
            X (pd.DataFrame): Features to pass through.

        Returns:
            pd.DataFrame: The same features.
        """
        return X

    def fit_transform(
        self, X: pd.DataFrame, y: pd.Series | None = None, **fit_params
    ) -> pd.DataFrame:
        """Records the call and returns the input unchanged.

        Args:
            X (pd.DataFrame): Features to pass through.
            y (pd.Series | None): Target, unused. Defaults to None.
            **fit_params: Ignored.

        Returns:
            pd.DataFrame: The same features.
        """
        FIT_TRANSFORM_CALLS.append("fit_transform")
        self.fitted_ = True
        return X


@pytest.fixture
def hyperparam_classification_data() -> tuple[pd.DataFrame, pd.Series]:
    """Provides a binary classification dataset for hyperparameter tuning testing.

    Returns:
        tuple[pd.DataFrame, pd.Series]: Feature matrix and binary target vector.
    """
    x_arr, y_arr = make_classification(
        n_samples=180,
        n_features=5,
        n_informative=3,
        n_redundant=1,
        random_state=21,
    )
    x_df = pd.DataFrame(x_arr, columns=[f"col_{i}" for i in range(5)])
    return x_df, pd.Series(y_arr, name="target")


@pytest.fixture
def hyperparam_cv_trainer(test_settings) -> CrossValidationTrainer:
    """Provides a CrossValidationTrainer configured for hyperparameter tuning testing.

    Args:
        test_settings: Test configuration fixture.

    Returns:
        CrossValidationTrainer: Configured trainer with a stratified 3-fold splitter.
    """
    return CrossValidationTrainer(
        metric_name="Accuracy",
        problem_type="classification",
        cv=StratifiedKFold(n_splits=3, shuffle=True, random_state=test_settings.SEED),
        verbose=False,
    )


@pytest.fixture
def real_optuna_trial() -> optuna.trial.Trial:
    """Provides a real Optuna trial obtained from a local study via ask().

    Returns:
        optuna.trial.Trial: A live trial usable with suggest_* methods.
    """
    study = optuna.create_study()
    return study.ask()


def test_randomsearchcv_tune_uses_holdout_when_valid_data_provided(
    hyperparam_classification_data, logistic_regression_estimator, hyperparam_cv_trainer
):
    """Tests tune() takes the hold-out branch when X_valid/y_valid are supplied.

    Args:
        hyperparam_classification_data (tuple): Feature matrix and binary target vector.
        logistic_regression_estimator (LogisticRegression): Configured estimator.
        hyperparam_cv_trainer (CrossValidationTrainer): Configured CV trainer.

    Asserts:
        - tune() returns an optuna Study with a completed best trial
        - The searched hyperparameter appears in best_params
    """
    X, y = hyperparam_classification_data
    X_train, X_valid, y_train, y_valid = train_test_split(
        X, y, test_size=0.3, random_state=17, stratify=y
    )
    tuner = RandomSearchCV(cross_validator=hyperparam_cv_trainer, n_trials=3, seed=17)
    params_grid = {"C": ("float", [0.01, 1.0], {"log": True})}

    study = tuner.tune(
        logistic_regression_estimator,
        X_train,
        y_train,
        params_grid,
        X_valid=X_valid,
        y_valid=y_valid,
    )

    if study.best_value is None:
        raise AssertionError()
    if "C" not in study.best_params:
        raise AssertionError()


def test_randomsearchcv_tune_uses_cross_validation_when_no_valid_data(
    hyperparam_classification_data, logistic_regression_estimator, hyperparam_cv_trainer
):
    """Tests tune() takes the CV branch when no hold-out data is supplied.

    Args:
        hyperparam_classification_data (tuple): Feature matrix and binary target vector.
        logistic_regression_estimator (LogisticRegression): Configured estimator.
        hyperparam_cv_trainer (CrossValidationTrainer): Configured CV trainer.

    Asserts:
        - tune() returns an optuna Study with a completed best trial
        - The searched hyperparameter appears in best_params
    """
    X, y = hyperparam_classification_data
    tuner = RandomSearchCV(cross_validator=hyperparam_cv_trainer, n_trials=3, seed=17)
    params_grid = {"C": ("float", [0.01, 1.0], {"log": True})}

    study = tuner.tune(logistic_regression_estimator, X, y, params_grid)

    if study.best_value is None:
        raise AssertionError()
    if "C" not in study.best_params:
        raise AssertionError()


def test_randomsearchcv_tune_runs_single_stage_on_small_data(
    hyperparam_classification_data, logistic_regression_estimator, hyperparam_cv_trainer
):
    """Tests tune() skips the explore stage when the dataset is small enough for full fidelity.

    Args:
        hyperparam_classification_data (tuple): Feature matrix and binary target vector.
        logistic_regression_estimator (LogisticRegression): Configured estimator.
        hyperparam_cv_trainer (CrossValidationTrainer): Configured CV trainer.

    Asserts:
        - No explore study is created for a dataset below the explore threshold
        - tune() still returns a Study with a valid best_value
    """
    X, y = hyperparam_classification_data
    tuner = RandomSearchCV(cross_validator=hyperparam_cv_trainer, n_trials=3, seed=17)
    params_grid = {"C": ("float", [0.01, 1.0], {"log": True})}

    study = tuner.tune(logistic_regression_estimator, X, y, params_grid)

    if tuner.explore_study_ is not None:
        raise AssertionError()
    if study.best_value is None:
        raise AssertionError()


def test_randomsearchcv_tune_runs_two_stage_when_data_exceeds_explore_threshold(
    hyperparam_classification_data, logistic_regression_estimator, hyperparam_cv_trainer
):
    """Tests tune() ranks on a subsample and confirms the shortlist at full fidelity.

    Args:
        hyperparam_classification_data (tuple): Feature matrix and binary target vector.
        logistic_regression_estimator (LogisticRegression): Configured estimator.
        hyperparam_cv_trainer (CrossValidationTrainer): Configured CV trainer.

    Asserts:
        - An explore study is created and runs the requested number of trials
        - The confirm study runs exactly top_k trials, one per shortlisted candidate
        - best_params_ is populated with the searched hyperparameter
    """
    X, y = hyperparam_classification_data
    tuner = RandomSearchCV(
        cross_validator=hyperparam_cv_trainer,
        n_trials=4,
        seed=17,
        explore_fraction=0.5,
        explore_min_rows=20,
        top_k=2,
    )
    params_grid = {"C": ("float", [0.01, 1.0], {"log": True})}

    study = tuner.tune(logistic_regression_estimator, X, y, params_grid)

    if tuner.explore_study_ is None:
        raise AssertionError()
    if len(tuner.explore_study_.trials) != 4:
        raise AssertionError()
    if len(study.trials) != 2:
        raise AssertionError()
    if "C" not in tuner.best_params_:
        raise AssertionError()


def test_randomsearchcv_tune_fits_preprocessing_once_per_fold_not_per_trial(
    hyperparam_classification_data, logistic_regression_estimator, test_settings
):
    """Tests the preprocessing pipeline is fitted once per fold and reused across trials.

    Args:
        hyperparam_classification_data (tuple): Feature matrix and binary target vector.
        logistic_regression_estimator (LogisticRegression): Configured estimator.
        test_settings: Test configuration fixture.

    Asserts:
        - fit_transform is called exactly n_splits times regardless of the trial count
    """
    X, y = hyperparam_classification_data
    FIT_TRANSFORM_CALLS.clear()
    trainer = CrossValidationTrainer(
        metric_name="Accuracy",
        problem_type="classification",
        cv=StratifiedKFold(n_splits=3, shuffle=True, random_state=test_settings.SEED),
        preprocessing_pipeline=Pipeline([("passthrough", CountingPassthrough())]),
        verbose=False,
    )
    tuner = RandomSearchCV(cross_validator=trainer, n_trials=5, seed=17, verbose=False)
    params_grid = {"C": ("float", [0.01, 1.0], {"log": True})}

    tuner.tune(logistic_regression_estimator, X, y, params_grid)

    if len(FIT_TRANSFORM_CALLS) != 3:
        raise AssertionError()


def test_randomsearchcv_tune_evaluates_estimator_defaults_as_first_trial(
    hyperparam_classification_data, logistic_regression_estimator, hyperparam_cv_trainer
):
    """Tests the estimator's own defaults are enqueued as the very first trial.

    Args:
        hyperparam_classification_data (tuple): Feature matrix and binary target vector.
        logistic_regression_estimator (LogisticRegression): Configured estimator.
        hyperparam_cv_trainer (CrossValidationTrainer): Configured CV trainer.

    Asserts:
        - Trial 0 uses the estimator's current C rather than a sampled value
    """
    X, y = hyperparam_classification_data
    tuner = RandomSearchCV(cross_validator=hyperparam_cv_trainer, n_trials=3, seed=17)
    params_grid = {"C": ("float", [0.01, 1.0], {"log": True})}

    study = tuner.tune(logistic_regression_estimator, X, y, params_grid)

    expected = logistic_regression_estimator.get_params()["C"]
    if study.trials[0].params["C"] != expected:
        raise AssertionError()


def test_randomsearchcv_tune_keeps_defaults_when_search_gains_are_noise_sized(
    hyperparam_classification_data, logistic_regression_estimator, hyperparam_cv_trainer
):
    """Tests the baseline guard returns the defaults when the search cannot beat them.

    Uses a grid with no tunable dimension, so every trial reproduces the baseline and
    no trial can clear the one-standard-error margin.

    Args:
        hyperparam_classification_data (tuple): Feature matrix and binary target vector.
        logistic_regression_estimator (LogisticRegression): Configured estimator.
        hyperparam_cv_trainer (CrossValidationTrainer): Configured CV trainer.

    Asserts:
        - selection_source_ reports the baseline rather than the searched result
        - best_params_ carries the constant value from the grid
    """
    X, y = hyperparam_classification_data
    tuner = RandomSearchCV(cross_validator=hyperparam_cv_trainer, n_trials=2, seed=17)
    params_grid = {"C": ("constant", [0.5])}

    tuner.tune(logistic_regression_estimator, X, y, params_grid)

    if tuner.selection_source_ != "baseline":
        raise AssertionError()
    if tuner.best_params_["C"] != 0.5:
        raise AssertionError()


def test_randomsearchcv_tune_returns_refit_ready_n_estimators_without_early_stopping(
    hyperparam_classification_data, test_settings
):
    """Tests best_params_ carries a concrete tree count and drops early-stopping keys.

    Args:
        hyperparam_classification_data (tuple): Feature matrix and binary target vector.
        test_settings: Test configuration fixture.

    Asserts:
        - n_estimators is present and positive in best_params_
        - No early-stopping key survives into best_params_
        - Per-fold best iterations are exposed for inspection
    """
    X, y = hyperparam_classification_data
    trainer = CrossValidationTrainer(
        metric_name="Accuracy",
        problem_type="classification",
        cv=StratifiedKFold(n_splits=3, shuffle=True, random_state=test_settings.SEED),
        verbose=False,
    )
    tuner = RandomSearchCV(cross_validator=trainer, n_trials=2, seed=17, verbose=False)
    estimator = LGBMClassifier(
        n_estimators=60, early_stopping_round=10, verbose=-1, random_state=17
    )
    params_grid = {
        "verbose": ("constant", [-1]),
        "num_leaves": ("int", [4, 16]),
    }

    tuner.tune(estimator, X, y, params_grid)

    if tuner.best_params_.get("n_estimators", 0) <= 0:
        raise AssertionError()
    if "early_stopping_round" in tuner.best_params_:
        raise AssertionError()
    if not tuner.best_iterations_:
        raise AssertionError()


def test_randomsearchcv_tune_supports_regression_problems():
    """Tests tune() completes on a regression problem with a minimize-direction metric.

    Asserts:
        - selection_source_ reports one of the two possible sources
        - best_params_ carries the searched hyperparameter
    """
    x_arr, y_arr = make_regression(
        n_samples=150, n_features=5, n_informative=3, random_state=17
    )
    X = pd.DataFrame(x_arr, columns=[f"col_{i}" for i in range(5)])
    y = pd.Series(y_arr, name="target")
    trainer = CrossValidationTrainer(
        metric_name="RMSE",
        problem_type="regression",
        cv=KFold(n_splits=3, shuffle=True, random_state=17),
        verbose=False,
    )
    tuner = RandomSearchCV(cross_validator=trainer, n_trials=3, seed=17, verbose=False)
    params_grid = {
        "verbose": ("constant", [-1]),
        "num_leaves": ("int", [4, 16]),
    }

    estimator = LGBMRegressor(n_estimators=30, verbose=-1, random_state=17)

    tuner.tune(estimator, X, y, params_grid)

    if tuner.selection_source_ not in {"tuned", "baseline"}:
        raise AssertionError()
    if "num_leaves" not in tuner.best_params_:
        raise AssertionError()


@pytest.mark.parametrize(
    ("param_type", "values", "kwargs", "value", "expected"),
    [
        ("float", [1e-3, 50.0], {"log": True}, 0.0, 1e-3),
        ("float", [0.5, 1.0], {}, 2.0, 1.0),
        ("int", [8, 256], {"step": 1}, 31, 31),
        ("int", [20, 5000], {"log": True}, 5, 20),
        ("int", [20, 5000], {"log": True}, 999999, 5000),
        ("int", [8, 256], {"step": 1}, 1000, 256),
        ("categorical", [True, False], {}, False, False),
        ("categorical", ["a", "b"], {}, "zzz", None),
    ],
)
def test_randomsearchcv_snap_to_space_clamps_defaults_into_the_grid(
    param_type, values, kwargs, value, expected
):
    """Tests estimator defaults outside the grid are clamped so they can be enqueued.

    Args:
        param_type (str): Parameter type under test.
        values (list): Bounds or choices for the parameter.
        kwargs (dict): Extra keyword arguments such as step or log.
        value: The estimator's current value.
        expected: Value expected after snapping.

    Asserts:
        - Out-of-range numeric defaults are clamped to the nearest bound
        - Unrepresentable categorical defaults return None
    """
    result = RandomSearchCV._snap_to_space(param_type, values, kwargs, value)

    if result != expected:
        raise AssertionError()


def test_randomsearchcv_early_stopping_split_never_overlaps_the_scored_fold(
    hyperparam_classification_data, hyperparam_cv_trainer
):
    """Tests each fold early-stops on rows that are absent from the rows it is scored on.

    This is the invariant the rewrite exists to restore: scoring a model on the same
    split that selected its stopping point rewards whichever configuration exploits
    that signal hardest, which is exactly the configuration that collapses on a
    fixed-iteration refit.

    Args:
        hyperparam_classification_data (tuple): Feature matrix and binary target vector.
        hyperparam_cv_trainer (CrossValidationTrainer): Configured CV trainer.

    Asserts:
        - Every fold carves a non-empty early-stopping split
        - The fit, early-stopping, and validation splits are pairwise disjoint
    """
    X, y = hyperparam_classification_data
    tuner = RandomSearchCV(cross_validator=hyperparam_cv_trainer, n_trials=1, seed=17)

    plan = tuner._build_plan(X, y, hyperparam_cv_trainer.cv, "confirm")

    if plan.cached is None:
        raise AssertionError()
    for fold in plan.cached:
        if fold.x_es is None or len(fold.x_es) == 0:
            raise AssertionError()
        fit_rows = set(fold.x_fit.index)
        es_rows = set(fold.x_es.index)
        valid_rows = set(fold.x_valid.index)
        if fit_rows & es_rows:
            raise AssertionError()
        if es_rows & valid_rows:
            raise AssertionError()
        if fit_rows & valid_rows:
            raise AssertionError()


def test_randomsearchcv_scale_row_params_rescales_counts_to_the_training_fidelity(
    hyperparam_cv_trainer,
):
    """Tests row-count parameters are rescaled when the explore stage trains on fewer rows.

    Args:
        hyperparam_cv_trainer (CrossValidationTrainer): Configured CV trainer.

    Asserts:
        - Integer counts scale by the fidelity fraction and stay integers
        - Floating hessian floors scale too
        - Full fidelity leaves every parameter untouched
        - Unrelated parameters are never touched
    """
    tuner = RandomSearchCV(cross_validator=hyperparam_cv_trainer, n_trials=1, seed=17)
    params = {"min_child_samples": 1000, "min_child_weight": 8.0, "num_leaves": 256}

    scaled = tuner._scale_row_params(params, 0.25)

    if scaled["min_child_samples"] != 250:
        raise AssertionError()
    if abs(scaled["min_child_weight"] - 2.0) > 1e-9:
        raise AssertionError()
    if scaled["num_leaves"] != 256:
        raise AssertionError()
    if tuner._scale_row_params(params, 1.0) != params:
        raise AssertionError()
    if tuner._scale_row_params({"min_child_samples": 2}, 0.01)["min_child_samples"] < 1:
        raise AssertionError()


def test_randomsearchcv_explore_fidelity_accounts_for_rows_and_fold_count(
    hyperparam_classification_data, hyperparam_cv_trainer
):
    """Tests the explore fidelity combines the subsample fraction with the fold counts.

    Args:
        hyperparam_classification_data (tuple): Feature matrix and binary target vector.
        hyperparam_cv_trainer (CrossValidationTrainer): 3-fold trainer used as confirm CV.

    Asserts:
        - A half-size subsample on 5 folds against the full data on 3 folds yields the
          ratio of their average training-fold sizes
        - A plan compared against itself yields exactly 1.0
    """
    X, y = hyperparam_classification_data
    tuner = RandomSearchCV(cross_validator=hyperparam_cv_trainer, n_trials=1, seed=17)
    confirm_plan = tuner._build_plan(X, y, hyperparam_cv_trainer.cv, "confirm")
    explore_plan = tuner._build_plan(
        X.iloc[:90], y.iloc[:90], KFold(n_splits=5, shuffle=True, random_state=1), "explore"
    )

    fidelity = tuner._explore_fidelity(explore_plan, confirm_plan)

    expected = (90 * (1 - 1 / 5)) / (180 * (1 - 1 / 3))
    if abs(fidelity - expected) > 1e-9:
        raise AssertionError()
    if tuner._explore_fidelity(confirm_plan, confirm_plan) != 1.0:
        raise AssertionError()


def test_randomsearchcv_baseline_is_not_clamped_when_scored_directly(
    logistic_regression_estimator, hyperparam_cv_trainer
):
    """Tests the safety baseline keeps the estimator's real defaults when scored directly.

    Clamping a default into the grid would mean the guard compares against something
    other than the library defaults it exists to protect.

    Args:
        logistic_regression_estimator (LogisticRegression): Estimator with C=1.0.
        hyperparam_cv_trainer (CrossValidationTrainer): Configured CV trainer.

    Asserts:
        - snap=False returns the true default even when it sits outside the grid
        - snap=True clamps it into the grid so Optuna can enqueue it
    """
    tuner = RandomSearchCV(cross_validator=hyperparam_cv_trainer, n_trials=1, seed=17)
    tuner.params_grid = {"C": ("float", [0.01, 0.5], {"log": True})}

    unsnapped = tuner._baseline_params(logistic_regression_estimator, snap=False)
    snapped = tuner._baseline_params(logistic_regression_estimator, snap=True)

    if unsnapped["C"] != logistic_regression_estimator.get_params()["C"]:
        raise AssertionError()
    if snapped["C"] != 0.5:
        raise AssertionError()


def test_randomsearchcv_required_margin_is_the_paired_standard_error():
    """Tests the guard margin is one standard error of the paired per-fold differences.

    Asserts:
        - The margin matches std(ddof=1)/sqrt(n) of the per-fold differences
        - A shift that is identical on every fold needs no margin at all
        - The margin is zero with fewer than two folds or mismatched fold counts
    """
    challenger = [0.53, 0.59, 0.61]
    reference = [0.55, 0.60, 0.64]
    differences = np.subtract(challenger, reference)
    expected = float(np.std(differences, ddof=1) / np.sqrt(len(differences)))

    if abs(RandomSearchCV._required_margin(challenger, reference) - expected) > 1e-12:
        raise AssertionError()
    if RandomSearchCV._required_margin([0.5, 0.6, 0.7], [0.52, 0.62, 0.72]) > 1e-12:
        raise AssertionError()
    if RandomSearchCV._required_margin([0.8], [0.7]) != 0.0:
        raise AssertionError()
    if RandomSearchCV._required_margin([0.8, 0.7], [0.7]) != 0.0:
        raise AssertionError()


def test_randomsearchcv_guard_accepts_a_consistent_gain_across_uneven_folds(
    hyperparam_cv_trainer,
):
    """Tests fold difficulty no longer masks a searched config that wins on every fold.

    Reproduces a 3-fold log-loss run whose folds differ widely in difficulty: the search
    is better on every fold, yet an unpaired margin built from the baseline's own fold
    spread exceeded the whole gain and kept the defaults.

    Args:
        hyperparam_cv_trainer (CrossValidationTrainer): Configured CV trainer.

    Asserts:
        - The searched configuration is selected.
    """
    tuner = _tuner_with_grid(hyperparam_cv_trainer, {"C": ("float", [0.01, 1.0])})
    tuner.metric_direction = "minimize"
    baseline_folds = [0.52, 0.58, 0.65]
    tuner._evaluate_config = lambda *_args, **_kwargs: (baseline_folds, [])
    tuner._penalized_score = lambda _scores: 0.583
    searched = _frozen_trial(0.559, [0.50, 0.555, 0.622], [100, 120, 140])
    plan = Mock(train_rows=100.0, refit_rows=150)

    tuner._select_best(plan, LGBMClassifier(), _study_of([searched], direction="minimize"))

    if tuner.selection_source_ != "tuned":
        raise AssertionError()


def test_fold_median_pruner_is_not_shielded_by_an_easy_first_fold():
    """Tests pruning compares the running mean at the current fold, not the best step.

    Args: None.

    Asserts:
        - A trial worse than the median at fold 2 is pruned although its fold-1 value is low
        - A trial better than the median survives
        - The last fold is never pruned
    """
    study = optuna.create_study(direction="minimize")
    for fold_1, fold_2 in [(0.40, 0.55), (0.41, 0.56), (0.42, 0.57)]:
        study.add_trial(
            optuna.trial.create_trial(
                value=fold_2,
                intermediate_values={0: fold_1, 1: fold_2, 2: fold_2},
            )
        )
    pruner = _FoldMedianPruner(n_folds=3, n_startup_trials=3)

    def running(values):
        return optuna.trial.create_trial(
            state=optuna.trial.TrialState.RUNNING,
            intermediate_values=dict(enumerate(values)),
        )

    if not pruner.prune(study, running([0.39, 0.60])):
        raise AssertionError()
    if pruner.prune(study, running([0.39, 0.50])):
        raise AssertionError()
    if pruner.prune(study, running([0.39, 0.60, 0.90])):
        raise AssertionError()


def test_randomsearchcv_single_stage_reports_saturated_grid_bounds(
    hyperparam_classification_data, logistic_regression_estimator, hyperparam_cv_trainer
):
    """Tests a single-stage search still reports bounds its best configs pressed against.

    Args:
        hyperparam_classification_data (tuple): Feature matrix and binary target vector.
        logistic_regression_estimator (LogisticRegression): Configured estimator.
        hyperparam_cv_trainer (CrossValidationTrainer): Configured CV trainer.

    Asserts:
        - A grid whose whole range sits far below the useful C values saturates high.
    """
    X, y = hyperparam_classification_data
    tuner = RandomSearchCV(
        cross_validator=hyperparam_cv_trainer, n_trials=12, seed=17, verbose=False, top_k=3
    )

    tuner.tune(logistic_regression_estimator, X, y, {"C": ("float", [1e-6, 1e-3], {"log": True})})

    if tuner.saturated_params_.get("C") != "high":
        raise AssertionError(tuner.saturated_params_)


@pytest.mark.parametrize(
    ("param_type", "param_value", "param_kwargs", "expected_type"),
    [
        ("int", [1, 10], {}, int),
        ("int", [20, 5000], {"log": True}, int),
        ("float", [0.01, 1.0], {"log": True}, float),
        ("float", [0.0, 1.0], {"step": 0.1}, float),
        ("categorical", ["a", "b", "c"], {}, str),
    ],
)
def test_randomsearchcv_get_param_samples_expected_types(
    real_optuna_trial, param_type, param_value, param_kwargs, expected_type
):
    """Tests get_param samples values of the expected type for each supported param_type.

    Args:
        real_optuna_trial (optuna.trial.Trial): Live trial fixture.
        param_type (str): Parameter type under test.
        param_value (list): Bounds or choices for the parameter.
        param_kwargs (dict): Extra keyword arguments (log/step).
        expected_type (type): Expected Python type of the sampled value.

    Asserts:
        - The sampled value is an instance of the expected type
        - Numeric values fall within the requested bounds
    """
    tuner = RandomSearchCV(cross_validator=Mock(), n_trials=1, seed=17)

    value = tuner.get_param(
        real_optuna_trial, "param", (param_type, param_value, param_kwargs)
    )

    if not isinstance(value, expected_type):
        raise AssertionError()
    if param_type in {"int", "float"}:
        low, high = param_value
        if not low <= value <= high:
            raise AssertionError()
    else:
        if value not in param_value:
            raise AssertionError()


def test_randomsearchcv_get_param_returns_constant_without_sampling(real_optuna_trial):
    """Tests get_param returns the first element for a constant param_type.

    Args:
        real_optuna_trial (optuna.trial.Trial): Live trial fixture.

    Asserts:
        - The returned value equals the constant's sole list entry
    """
    tuner = RandomSearchCV(cross_validator=Mock(), n_trials=1, seed=17)

    value = tuner.get_param(real_optuna_trial, "param", ("constant", ["fixed_value"]))

    if value != "fixed_value":
        raise AssertionError()


def test_randomsearchcv_get_param_raises_error_for_unknown_param_type(
    real_optuna_trial,
):
    """Tests get_param raises ValueError for an unsupported param_type.

    Args:
        real_optuna_trial (optuna.trial.Trial): Live trial fixture.

    Asserts:
        - ValueError is raised mentioning the unknown param_type
    """
    tuner = RandomSearchCV(cross_validator=Mock(), n_trials=1, seed=17)

    with pytest.raises(ValueError, match="Unknown param_type"):
        tuner.get_param(real_optuna_trial, "param", ("weird_type", [1, 2]))


def test_randomsearchcv_tune_prints_trial_and_summary_lines_when_verbose(
    hyperparam_classification_data,
    logistic_regression_estimator,
    hyperparam_cv_trainer,
    capsys,
):
    """Tests tune() prints a start line, one line per trial, and a finish summary.

    Args:
        hyperparam_classification_data (tuple): Feature matrix and binary target vector.
        logistic_regression_estimator (LogisticRegression): Configured estimator.
        hyperparam_cv_trainer (CrossValidationTrainer): Configured CV trainer.
        capsys: Pytest fixture capturing stdout/stderr.

    Asserts:
        - The starting summary line is printed
        - At least one per-trial line per requested trial is printed
        - The final best-trial summary is printed
    """
    X, y = hyperparam_classification_data
    tuner = RandomSearchCV(
        cross_validator=hyperparam_cv_trainer, n_trials=3, seed=17, verbose=True
    )
    params_grid = {"C": ("float", [0.01, 1.0], {"log": True})}

    tuner.tune(logistic_regression_estimator, X, y, params_grid)

    out = capsys.readouterr().out
    if "Starting random search" not in out:
        raise AssertionError()
    if out.count("Trial ") < 3:
        raise AssertionError()
    if "Best" not in out:
        raise AssertionError()


def test_randomsearchcv_tune_is_silent_and_restores_verbosity_when_not_verbose(
    hyperparam_classification_data,
    logistic_regression_estimator,
    hyperparam_cv_trainer,
    capsys,
):
    """Tests tune() prints nothing and does not leak Optuna's global log verbosity.

    Args:
        hyperparam_classification_data (tuple): Feature matrix and binary target vector.
        logistic_regression_estimator (LogisticRegression): Configured estimator.
        hyperparam_cv_trainer (CrossValidationTrainer): Configured CV trainer.
        capsys: Pytest fixture capturing stdout/stderr.

    Asserts:
        - Nothing is printed to stdout or stderr during tune()
        - Optuna's global logging verbosity is restored to its pre-call value afterward
    """
    X, y = hyperparam_classification_data
    tuner = RandomSearchCV(
        cross_validator=hyperparam_cv_trainer, n_trials=3, seed=17, verbose=False
    )
    params_grid = {"C": ("float", [0.01, 1.0], {"log": True})}
    previous_verbosity = optuna.logging.get_verbosity()

    tuner.tune(logistic_regression_estimator, X, y, params_grid)

    captured = capsys.readouterr()
    if captured.out != "":
        raise AssertionError()
    if captured.err != "":
        raise AssertionError()
    if optuna.logging.get_verbosity() != previous_verbosity:
        raise AssertionError()


if __name__ == "__main__":
    print("Run this file with pytest to execute tests.")


def _tuner_with_grid(cv_trainer, params_grid, **config_kwargs) -> RandomSearchCV:
    """Builds a tuner whose grid is already installed, bypassing a full tune() run.

    Args:
        cv_trainer (CrossValidationTrainer): Configured CV trainer.
        params_grid (dict): Search space to install.
        **config_kwargs: Keyword-only overrides forwarded to RandomSearchCV.

    Returns:
        RandomSearchCV: Tuner ready for search-space helper calls.
    """
    tuner = RandomSearchCV(
        cross_validator=cv_trainer,
        n_trials=10,
        seed=17,
        verbose=False,
        **config_kwargs,
    )
    tuner.params_grid = params_grid
    return tuner


def _frozen_trial(value, fold_scores, best_iterations):
    """Builds a completed FrozenTrial carrying the user attrs the tuner reads.

    Args:
        value (float): Objective value.
        fold_scores (list[float]): Per-fold scores stored as a user attr.
        best_iterations (list[int]): Iteration counts stored as a user attr.

    Returns:
        optuna.trial.FrozenTrial: Trial usable inside a study.
    """
    return optuna.trial.create_trial(
        params={"C": 0.1},
        distributions={"C": optuna.distributions.FloatDistribution(0.01, 1.0)},
        value=value,
        user_attrs={
            "fold_scores": fold_scores,
            "best_iterations": best_iterations,
            "params": {"C": 0.1},
        },
    )


def _study_of(trials, direction="maximize") -> optuna.study.Study:
    """Creates a study preloaded with the supplied trials.

    Args:
        trials (list): Trials to add.
        direction (str): Optimisation direction. Defaults to "maximize".

    Returns:
        optuna.study.Study: Study containing the trials.
    """
    study = optuna.create_study(direction=direction)
    for trial in trials:
        study.add_trial(trial)
    return study


def test_randomsearchcv_narrow_space_shrinks_ranges_inside_the_original_grid(
    hyperparam_cv_trainer,
):
    """Tests the confirm-stage box tightens around the elite without leaving the grid.

    Args:
        hyperparam_cv_trainer (CrossValidationTrainer): Configured CV trainer.

    Asserts:
        - An unsaturated range narrows strictly inside the caller's bounds
        - Log, step, and integer typing survive narrowing
        - The narrowed low bound stays below the narrowed high bound
    """
    grid = {
        "n_estimators": ("constant", [100]),
        "learning_rate": ("float", [0.001, 1.0], {"log": True}),
        "max_depth": ("int", [2, 20], {"step": 2}),
    }
    tuner = _tuner_with_grid(hyperparam_cv_trainer, grid)
    elite = [
        {"learning_rate": 0.05, "max_depth": 8},
        {"learning_rate": 0.09, "max_depth": 10},
    ]

    space = tuner._narrow_space(elite)

    lr_type, lr_values, lr_kwargs = tuner._unpack_spec(space["learning_rate"])
    depth_type, depth_values, depth_kwargs = tuner._unpack_spec(space["max_depth"])
    if space["n_estimators"] != grid["n_estimators"]:
        raise AssertionError()
    if lr_type != "float" or not lr_kwargs.get("log"):
        raise AssertionError()
    if not 0.001 < lr_values[0] < lr_values[1] < 1.0:
        raise AssertionError()
    if depth_type != "int" or depth_kwargs.get("step") != 2:
        raise AssertionError()
    if not all(isinstance(bound, int) for bound in depth_values):
        raise AssertionError()
    if not 2 <= depth_values[0] < depth_values[1] <= 20:
        raise AssertionError()


def test_randomsearchcv_narrow_space_expands_only_the_saturated_bound(
    hyperparam_cv_trainer,
):
    """Tests a bound the elite presses against is pushed out by exactly one step.

    Args:
        hyperparam_cv_trainer (CrossValidationTrainer): Configured CV trainer.

    Asserts:
        - The saturated side is reported on saturated_params_
        - The saturated low bound drops below the caller's bound
        - The library hard limit caps the expansion
        - An unsaturated parameter in the same grid is not expanded
    """
    grid = {
        "num_leaves": ("int", [8, 512], {"log": True}),
        "learning_rate": ("float", [0.03, 0.2], {"log": True}),
        "subsample": ("float", [0.5, 1.0]),
    }
    tuner = _tuner_with_grid(hyperparam_cv_trainer, grid)
    elite = [
        {"num_leaves": 8, "learning_rate": 0.031, "subsample": 0.7},
        {"num_leaves": 9, "learning_rate": 0.033, "subsample": 0.75},
    ]

    space = tuner._narrow_space(elite)

    if tuner.saturated_params_.get("num_leaves") != "low":
        raise AssertionError()
    if tuner.saturated_params_.get("learning_rate") != "low":
        raise AssertionError()
    if "subsample" in tuner.saturated_params_:
        raise AssertionError()
    if space["num_leaves"][1][0] >= 8:
        raise AssertionError()
    if space["num_leaves"][1][0] < 2:
        raise AssertionError()
    if space["learning_rate"][1][0] >= 0.03:
        raise AssertionError()
    if space["subsample"][1][0] < 0.5:
        raise AssertionError()


def test_randomsearchcv_saturation_report_ignores_bounds_at_the_library_limit(
    hyperparam_cv_trainer,
):
    """Tests a bound with no room to expand into is never reported as saturated.

    Args:
        hyperparam_cv_trainer (CrossValidationTrainer): Configured CV trainer.

    Asserts:
        - An elite pressed against subsample's ceiling of 1.0 raises no report
        - An elite sitting mid-range raises no report
    """
    grid = {
        "subsample": ("float", [0.5, 1.0]),
        "colsample_bytree": ("float", [0.3, 1.0]),
    }
    tuner = _tuner_with_grid(hyperparam_cv_trainer, grid)
    elite = [
        {"subsample": 0.99, "colsample_bytree": 0.64},
        {"subsample": 1.0, "colsample_bytree": 0.66},
    ]

    report = tuner._saturation_report(elite)

    if report:
        raise AssertionError()


def test_randomsearchcv_narrow_space_keeps_a_valid_range_for_a_degenerate_elite(
    hyperparam_cv_trainer,
):
    """Tests an elite that agreed on one value still yields a searchable range.

    Args:
        hyperparam_cv_trainer (CrossValidationTrainer): Configured CV trainer.

    Asserts:
        - Every narrowed numeric range keeps low strictly below high
    """
    grid = {
        "learning_rate": ("float", [0.01, 1.0], {"log": True}),
        "max_depth": ("int", [2, 20], {"step": 2}),
        "gamma": ("float", [0.0, 5.0]),
    }
    tuner = _tuner_with_grid(hyperparam_cv_trainer, grid)
    elite = [{"learning_rate": 0.1, "max_depth": 6, "gamma": 1.0}] * 3

    space = tuner._narrow_space(elite)

    for name in grid:
        low, high = space[name][1]
        if not low < high:
            raise AssertionError()


def test_randomsearchcv_narrow_space_keeps_only_categories_the_elite_used(
    hyperparam_cv_trainer,
):
    """Tests categorical narrowing drops unused choices but never empties the space.

    Args:
        hyperparam_cv_trainer (CrossValidationTrainer): Configured CV trainer.

    Asserts:
        - Choices the elite used survive and unused ones are dropped
        - A single surviving choice falls back to the caller's full list
    """
    grid = {
        "booster": ("categorical", ["gbtree", "dart", "gblinear"]),
        "policy": ("categorical", ["depthwise", "lossguide"]),
    }
    tuner = _tuner_with_grid(hyperparam_cv_trainer, grid)
    elite = [
        {"booster": "gbtree", "policy": "depthwise"},
        {"booster": "dart", "policy": "depthwise"},
    ]

    space = tuner._narrow_space(elite)

    if space["booster"][1] != ["gbtree", "dart"]:
        raise AssertionError()
    if space["policy"][1] != ["depthwise", "lossguide"]:
        raise AssertionError()


def test_randomsearchcv_confirm_stage_searches_beyond_the_shortlist(
    hyperparam_classification_data, logistic_regression_estimator, hyperparam_cv_trainer
):
    """Tests the confirm stage samples new configurations instead of replaying the shortlist.

    Replaying only the shortlist made the confirm stage re-rank near-duplicates whose
    scores differed by less than the fold noise, so it must now search its own box.

    Args:
        hyperparam_classification_data (tuple): Feature matrix and binary target vector.
        logistic_regression_estimator (LogisticRegression): Configured estimator.
        hyperparam_cv_trainer (CrossValidationTrainer): Configured CV trainer.

    Asserts:
        - The confirm study runs max(top_k, n_trials // 4) trials, not top_k
        - At least one confirm trial differs from every warm-started candidate
    """
    X, y = hyperparam_classification_data
    tuner = RandomSearchCV(
        cross_validator=hyperparam_cv_trainer,
        n_trials=24,
        seed=17,
        verbose=False,
        explore_fraction=0.5,
        explore_min_rows=20,
        top_k=2,
    )

    study = tuner.tune(
        logistic_regression_estimator,
        X,
        y,
        {"C": ("float", [0.01, 1.0], {"log": True})},
    )

    if len(study.trials) != 6:
        raise AssertionError()
    warm_started = {round(t.params["C"], 12) for t in study.trials[:2]}
    fresh = [
        t for t in study.trials[2:] if round(t.params["C"], 12) not in warm_started
    ]
    if not fresh:
        raise AssertionError()


def test_randomsearchcv_confirm_selection_prefers_the_cheaper_of_tied_trials(
    hyperparam_cv_trainer,
):
    """Tests the one-standard-error rule breaks confirm ties toward the cheaper model.

    Args:
        hyperparam_cv_trainer (CrossValidationTrainer): Configured CV trainer.

    Asserts:
        - The tie count is recorded on confirm_ties_
        - The trial with fewer boosting rounds wins despite a lower objective value
    """
    tuner = _tuner_with_grid(hyperparam_cv_trainer, {"C": ("float", [0.01, 1.0])})
    tuner.metric_direction = "maximize"
    expensive = _frozen_trial(0.9000, [0.88, 0.90, 0.92], [900, 950, 1000])
    cheap = _frozen_trial(0.8999, [0.87, 0.91, 0.92], [90, 95, 100])
    study = _study_of([expensive, cheap])

    chosen = tuner._select_confirm_trial(study.trials)

    if tuner.confirm_ties_ != 2:
        raise AssertionError()
    if chosen.user_attrs["best_iterations"] != [90, 95, 100]:
        raise AssertionError()


def test_randomsearchcv_confirm_selection_keeps_argmax_when_scores_differ(
    hyperparam_cv_trainer,
):
    """Tests a genuine improvement beyond one standard error is not traded for cheapness.

    Args:
        hyperparam_cv_trainer (CrossValidationTrainer): Configured CV trainer.

    Asserts:
        - The higher-scoring trial wins when it clears the standard-error margin
        - Only that trial is counted as tied
    """
    tuner = _tuner_with_grid(hyperparam_cv_trainer, {"C": ("float", [0.01, 1.0])})
    tuner.metric_direction = "maximize"
    strong = _frozen_trial(0.95, [0.949, 0.950, 0.951], [900, 950, 1000])
    cheap = _frozen_trial(0.80, [0.799, 0.800, 0.801], [10, 12, 14])
    study = _study_of([strong, cheap])

    chosen = tuner._select_confirm_trial(study.trials)

    if chosen.value != strong.value:
        raise AssertionError()
    if tuner.confirm_ties_ != 1:
        raise AssertionError()


def test_randomsearchcv_detects_iteration_censoring_at_the_round_ceiling(
    hyperparam_cv_trainer,
):
    """Tests folds that exhaust the boosting-round ceiling are flagged as censored.

    Args:
        hyperparam_cv_trainer (CrossValidationTrainer): Configured CV trainer.

    Asserts:
        - Iterations reaching the ceiling set n_estimators_censored_
        - Iterations stopping well short of the ceiling leave it False
        - A tree budget is still returned in both cases
    """
    tuner = _tuner_with_grid(
        hyperparam_cv_trainer, {"n_estimators": ("constant", [4000])}
    )
    estimator = LGBMClassifier(n_estimators=4000, verbose=-1)

    censored = tuner._resolve_n_estimators(
        estimator, [3998, 3996, 3982], {"n_estimators": 4000}, 1.25
    )
    if not tuner.n_estimators_censored_:
        raise AssertionError()
    if "n_estimators" not in censored:
        raise AssertionError()

    converged = tuner._resolve_n_estimators(
        estimator, [412, 388, 401], {"n_estimators": 4000}, 1.25
    )
    if tuner.n_estimators_censored_:
        raise AssertionError()
    if "n_estimators" not in converged:
        raise AssertionError()


def test_randomsearchcv_reconciles_categorical_columns_the_pipeline_drops(
    hyperparam_cv_trainer,
):
    """Tests declared categoricals absent from the transformed frame are dropped.

    A target-encoding pipeline replaces the raw categorical columns, so a cat_features
    list written against the untransformed frame names columns the estimator never sees.

    Args:
        hyperparam_cv_trainer (CrossValidationTrainer): Configured CV trainer.

    Asserts:
        - Surviving columns are kept and missing ones removed
        - Other parameters are left untouched
        - The caller's own dictionary is not mutated
    """
    tuner = _tuner_with_grid(hyperparam_cv_trainer, {})
    tuner._remember_transformed_columns(
        pd.DataFrame(
            {
                "RACE": pd.Categorical(["MONZA", "SPA"]),
                "RACE_TARGET_ENCODED": [0.4, 0.6],
                "LAP": [1.0, 2.0],
            }
        )
    )
    params = {"cat_features": ["RACE", "COMPOUND"], "depth": 6}

    reconciled = tuner._reconcile_categorical_params(params)

    if reconciled["cat_features"] != ["RACE"]:
        raise AssertionError()
    if reconciled["depth"] != 6:
        raise AssertionError()
    if params["cat_features"] != ["RACE", "COMPOUND"]:
        raise AssertionError()


def test_randomsearchcv_falls_back_to_frame_dtypes_when_no_declaration_survives(
    hyperparam_cv_trainer,
):
    """Tests a wholly stale declaration is replaced by the frame's own categoricals.

    Raising here would block the search over a declaration the tuner can repair: the
    transformed frame states exactly which columns need categorical treatment.

    Args:
        hyperparam_cv_trainer (CrossValidationTrainer): Configured CV trainer.

    Asserts:
        - The frame's non-numeric columns replace the stale declaration
        - An all-numeric frame yields an empty declaration rather than an error
    """
    tuner = _tuner_with_grid(hyperparam_cv_trainer, {})
    tuner._remember_transformed_columns(
        pd.DataFrame({"CIRCUIT": pd.Categorical(["MONZA", "SPA"]), "LAP": [1.0, 2.0]})
    )

    reconciled = tuner._reconcile_categorical_params(
        {"cat_features": ["RACE", "COMPOUND"]}
    )
    if reconciled["cat_features"] != ["CIRCUIT"]:
        raise AssertionError()

    tuner._remember_transformed_columns(
        pd.DataFrame({"RACE_TARGET_ENCODED": [0.4, 0.6], "LAP": [1.0, 2.0]})
    )
    encoded_away = tuner._reconcile_categorical_params(
        {"cat_features": ["RACE", "COMPOUND"]}
    )
    if encoded_away["cat_features"] != []:
        raise AssertionError()


def test_randomsearchcv_tune_runs_trials_in_parallel(
    hyperparam_classification_data, hyperparam_cv_trainer
):
    """Tests tune() completes with concurrent trials and stays inside the grid.

    Args:
        hyperparam_classification_data (tuple): Feature matrix and binary target vector.
        hyperparam_cv_trainer (CrossValidationTrainer): Configured CV trainer.

    Asserts:
        - Every requested trial finishes (none fail under concurrency).
        - The sampler uses a constant liar for trials still running.
        - The tuned learning rate lies inside the searched bounds.
    """
    X, y = hyperparam_classification_data
    tuner = RandomSearchCV(
        cross_validator=hyperparam_cv_trainer,
        n_trials=8,
        seed=17,
        n_parallel_trials=4,
        es_holdout_size=0.0,
    )
    params_grid = {
        "n_estimators": ("constant", [20]),
        "verbose": ("constant", [-1]),
        "n_jobs": ("constant", [1]),
        "learning_rate": ("float", [0.05, 0.3], {"log": True}),
    }
    estimator = LGBMClassifier(n_estimators=20, verbose=-1, n_jobs=1, learning_rate=0.1)

    study = tuner.tune(estimator, X, y, params_grid)

    finished = [t for t in study.trials if t.state.is_finished()]
    if any(t.state == optuna.trial.TrialState.FAIL for t in finished):
        raise AssertionError()
    if len(finished) != len(study.trials):
        raise AssertionError()
    if study.sampler._constant_liar is not True:
        raise AssertionError()
    if not 0.05 <= tuner.best_params_["learning_rate"] <= 0.3:
        raise AssertionError()


@pytest.mark.parametrize(
    ("option", "value"),
    [
        ("n_parallel_trials", 0),
        ("top_k", 0),
        ("explore_fraction", 0.0),
        ("explore_fraction", 1.5),
        ("es_holdout_size", 1.0),
        ("es_holdout_size", -0.1),
    ],
)
def test_randomsearchcv_rejects_invalid_staging_options(option, value):
    """Tests the constructor rejects staging options that cannot run.

    Args:
        option (str): Keyword-only constructor option under test.
        value (float | int): Invalid value for that option.

    Asserts:
        - ValueError is raised naming the offending option.
    """
    with pytest.raises(ValueError, match=option):
        RandomSearchCV(cross_validator=Mock(), **{option: value})


def test_randomsearchcv_prints_parallel_trials_in_trial_number_order(
    hyperparam_classification_data, hyperparam_cv_trainer, capsys
):
    """Tests trial lines come out in trial-number order even when trials run in parallel.

    Args:
        hyperparam_classification_data (tuple): Feature matrix and binary target vector.
        hyperparam_cv_trainer (CrossValidationTrainer): Configured CV trainer.
        capsys: Pytest fixture capturing stdout/stderr.

    Asserts:
        - Every trial is printed exactly once, numbered 1..n_trials in order.
    """
    X, y = hyperparam_classification_data
    n_trials = 12
    tuner = RandomSearchCV(
        cross_validator=hyperparam_cv_trainer, n_trials=n_trials, seed=17, n_parallel_trials=4
    )
    params_grid = {
        "n_estimators": ("constant", [20]),
        "verbose": ("constant", [-1]),
        "n_jobs": ("constant", [1]),
        "learning_rate": ("float", [0.05, 0.3], {"log": True}),
    }
    estimator = LGBMClassifier(n_estimators=20, verbose=-1, n_jobs=1, learning_rate=0.1)

    tuner.tune(estimator, X, y, params_grid)

    printed = [
        int(line.split("Trial ")[1].split("/")[0])
        for line in capsys.readouterr().out.splitlines()
        if "] Trial " in line
    ]
    if printed != list(range(1, n_trials + 1)):
        raise AssertionError(printed)


def test_randomsearchcv_tie_break_never_hands_the_win_back_to_the_baseline(
    hyperparam_cv_trainer,
):
    """Tests the one-standard-error tie rule only chooses among trials beating the baseline.

    The cheap trial ties with the best one, but on its own it does not clear the baseline
    guard. Breaking the tie before applying the guard would pick the cheap trial and then
    reject it, returning the baseline although a searched trial genuinely beat it.

    Args:
        hyperparam_cv_trainer (CrossValidationTrainer): Configured CV trainer.

    Asserts:
        - The best trial is selected and the source is "tuned".
    """
    tuner = _tuner_with_grid(hyperparam_cv_trainer, {"C": ("float", [0.01, 1.0])})
    tuner.metric_direction = "maximize"
    tuner._penalized_score = lambda _scores: 0.895
    tuner._evaluate_config = lambda *_args, **_kwargs: ([0.89, 0.895, 0.90], [])
    best = _frozen_trial(0.905, [0.90, 0.905, 0.91], [900, 950, 1000])
    cheap = _frozen_trial(0.903, [0.88, 0.905, 0.93], [90, 95, 100])
    plan = Mock(train_rows=100.0, refit_rows=150)

    tuner._select_best(plan, LGBMClassifier(), _study_of([best, cheap]))

    if tuner.selection_source_ != "tuned":
        raise AssertionError()
    if tuner.best_score_ != 0.905:
        raise AssertionError()


def test_randomsearchcv_holdout_refit_scale_ignores_the_cv_fold_count(
    hyperparam_classification_data, hyperparam_cv_trainer
):
    """Tests hold-out tuning scales boosting rounds only for the early-stopping carve-out.

    Args:
        hyperparam_classification_data (tuple): Feature matrix and binary target vector.
        hyperparam_cv_trainer (CrossValidationTrainer): 3-fold trainer, unused by hold-out.

    Asserts:
        - The refit scale equals training rows over rows left after the carve-out.
    """
    X, y = hyperparam_classification_data
    X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.25, random_state=17)
    tuner = RandomSearchCV(cross_validator=hyperparam_cv_trainer, n_trials=1, seed=17)

    plan = tuner._build_holdout_plan(X_train, y_train, X_valid, y_valid)

    expected = len(X_train) / (len(X_train) - int(len(X_train) * 0.1))
    if abs(tuner._refit_scale(plan) - expected) > 1e-9:
        raise AssertionError()
