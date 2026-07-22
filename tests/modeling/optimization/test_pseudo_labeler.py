"""Tests for kvbiii_ml.modeling.optimization.pseudo_labeler module."""

import os

os.environ["MPLBACKEND"] = "Agg"

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
from sklearn.model_selection import KFold

from kvbiii_ml.modeling.optimization.pseudo_labeler import PseudoLabelGenerator
from kvbiii_ml.modeling.training.cross_validation import CrossValidationTrainer

_N_FEATURES = 3
_FEATURE_NAMES = [f"feature_{i}" for i in range(_N_FEATURES)]


@pytest.fixture(autouse=True)
def _close_figures_after_test():
    """Closes all matplotlib figures after each test to avoid memory growth.

    Yields:
        None: Control back to the test before teardown.
    """
    yield
    plt.close("all")


@pytest.fixture
def pseudo_cv_trainer(kfold_cv: KFold) -> CrossValidationTrainer:
    """Provides a CrossValidationTrainer configured for pseudo-labeling tests.

    Args:
        kfold_cv (KFold): Shared cross-validation splitter fixture.

    Returns:
        CrossValidationTrainer: Configured trainer for classification.
    """
    return CrossValidationTrainer(
        problem_type="classification",
        metric_name="Accuracy",
        cv=kfold_cv,
        verbose=False,
    )


@pytest.fixture
def small_labeled_data() -> tuple[pd.DataFrame, pd.Series]:
    """Provides a small synthetic binary classification training dataset.

    Returns:
        tuple[pd.DataFrame, pd.Series]: Feature matrix and binary target vector.
    """
    rng = np.random.default_rng(17)
    X = pd.DataFrame(rng.normal(size=(40, _N_FEATURES)), columns=_FEATURE_NAMES)
    y = pd.Series((X["feature_0"] + rng.normal(scale=0.5, size=40) > 0).astype(int))
    return X, y


@pytest.fixture
def small_unlabeled_data() -> pd.DataFrame:
    """Provides a small synthetic unlabeled feature matrix.

    Returns:
        pd.DataFrame: Unlabeled feature matrix matching small_labeled_data columns.
    """
    rng = np.random.default_rng(99)
    return pd.DataFrame(rng.normal(size=(30, _N_FEATURES)), columns=_FEATURE_NAMES)


@pytest.fixture
def imbalanced_labeled_data() -> tuple[pd.DataFrame, pd.Series]:
    """Provides a small binary dataset with fewer than 10 true positives for class 1.

    Ones are placed at evenly spaced indices so each contiguous, non-shuffled KFold
    training split retains samples of both classes.

    Returns:
        tuple[pd.DataFrame, pd.Series]: Feature matrix and imbalanced target vector.
    """
    rng = np.random.default_rng(5)
    n_samples = 24
    y_values = np.zeros(n_samples, dtype=int)
    y_values[[0, 5, 10, 15, 20]] = 1
    X = pd.DataFrame(rng.normal(size=(n_samples, _N_FEATURES)), columns=_FEATURE_NAMES)
    X.loc[y_values == 1, "feature_0"] += 2.0
    return X, pd.Series(y_values)


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"threshold_method": "bogus"}, "threshold_method must be one of"),
        ({"threshold_percentile": -0.1}, "threshold_percentile must be in"),
        ({"threshold_percentile": 100.1}, "threshold_percentile must be in"),
        ({"top_k_pct": 0.0}, "top_k_pct must be in"),
        ({"top_k_pct": 1.1}, "top_k_pct must be in"),
        ({"fixed_threshold": 0.0}, "fixed_threshold must be in"),
        ({"fixed_threshold": 1.0}, "fixed_threshold must be in"),
    ],
)
def test_pseudolabelgenerator_init_raises_valueerror_for_invalid_configuration(
    kwargs, match, pseudo_cv_trainer
):
    """Tests PseudoLabelGenerator initialization rejects invalid configurations.

    Args:
        kwargs (dict): Constructor keyword arguments under test.
        match (str): Expected substring of the raised ValueError message.
        pseudo_cv_trainer (CrossValidationTrainer): Configured CV trainer.

    Asserts:
        - ValueError is raised for an unsupported threshold_method
        - ValueError is raised for threshold_percentile outside [0, 100]
        - ValueError is raised for top_k_pct outside (0, 1]
        - ValueError is raised for fixed_threshold outside (0, 1)
    """
    with pytest.raises(ValueError, match=match):
        PseudoLabelGenerator(cross_validator=pseudo_cv_trainer, **kwargs)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"threshold_method": "auto"},
        {"threshold_method": "top_k"},
        {"threshold_method": "fixed"},
        {"threshold_percentile": 0.0},
        {"threshold_percentile": 100.0},
        {"top_k_pct": 1.0},
        {"fixed_threshold": 0.01},
        {"fixed_threshold": 0.99},
    ],
)
def test_pseudolabelgenerator_init_accepts_valid_boundary_configuration(
    kwargs, pseudo_cv_trainer
):
    """Tests PseudoLabelGenerator initialization accepts valid boundary values.

    Args:
        kwargs (dict): Constructor keyword arguments under test.
        pseudo_cv_trainer (CrossValidationTrainer): Configured CV trainer.

    Asserts:
        - No exception is raised for any valid boundary configuration
        - The cross_validator is stored unchanged
    """
    generator = PseudoLabelGenerator(cross_validator=pseudo_cv_trainer, **kwargs)

    if generator.cross_validator is not pseudo_cv_trainer:
        raise AssertionError()


def test_pseudolabelgenerator_fit_auto_strategy_builds_pseudo_labels_dataframe(
    small_labeled_data, small_unlabeled_data, pseudo_cv_trainer, logistic_regression_estimator
):
    """Tests fit() with the "auto" threshold strategy produces a valid pseudo-labels frame.

    Args:
        small_labeled_data (tuple): Labeled feature matrix and target vector.
        small_unlabeled_data (pd.DataFrame): Unlabeled feature matrix.
        pseudo_cv_trainer (CrossValidationTrainer): Configured CV trainer.
        logistic_regression_estimator (LogisticRegression): Configured estimator.

    Asserts:
        - fit() returns self for chaining
        - pseudo_labels_df_ has the expected marker columns and feature columns
        - Every retained row has _selected == True
        - At most as many rows are retained as unlabeled samples were provided
        - Every retained row's confidence meets its class threshold
    """
    X_train, y_train = small_labeled_data
    generator = PseudoLabelGenerator(
        cross_validator=pseudo_cv_trainer, threshold_method="auto", threshold_percentile=70.0
    )

    fitted = generator.fit(
        logistic_regression_estimator, X_train, y_train, small_unlabeled_data
    )

    if fitted is not generator:
        raise AssertionError()
    expected_columns = {"_pseudo_label", "_confidence", "_selected", *X_train.columns}
    if not expected_columns.issubset(set(generator.pseudo_labels_df_.columns)):
        raise AssertionError()
    if not generator.pseudo_labels_df_["_selected"].all():
        raise AssertionError()
    if len(generator.pseudo_labels_df_) > len(small_unlabeled_data):
        raise AssertionError()
    confidences = generator.pseudo_labels_df_["_confidence"].to_numpy()
    thresholds_per_row = np.array(
        [generator.thresholds_[label] for label in generator.pseudo_labels_df_["_pseudo_label"]]
    )
    if not np.all(confidences >= thresholds_per_row - 1e-9):
        raise AssertionError()


def test_pseudolabelgenerator_fit_top_k_strategy_builds_pseudo_labels_dataframe(
    small_labeled_data, small_unlabeled_data, pseudo_cv_trainer, logistic_regression_estimator
):
    """Tests fit() with the "top_k" threshold strategy produces a valid pseudo-labels frame.

    Args:
        small_labeled_data (tuple): Labeled feature matrix and target vector.
        small_unlabeled_data (pd.DataFrame): Unlabeled feature matrix.
        pseudo_cv_trainer (CrossValidationTrainer): Configured CV trainer.
        logistic_regression_estimator (LogisticRegression): Configured estimator.

    Asserts:
        - pseudo_labels_df_ has the expected marker columns and feature columns
        - Every retained row's confidence meets its class threshold
        - Recomputed per-class thresholds match the generator's stored thresholds
    """
    X_train, y_train = small_labeled_data
    generator = PseudoLabelGenerator(
        cross_validator=pseudo_cv_trainer, threshold_method="top_k", top_k_pct=0.2
    )

    generator.fit(logistic_regression_estimator, X_train, y_train, small_unlabeled_data)

    expected_columns = {"_pseudo_label", "_confidence", "_selected", *X_train.columns}
    if not expected_columns.issubset(set(generator.pseudo_labels_df_.columns)):
        raise AssertionError()
    confidences = generator.pseudo_labels_df_["_confidence"].to_numpy()
    thresholds_per_row = np.array(
        [generator.thresholds_[label] for label in generator.pseudo_labels_df_["_pseudo_label"]]
    )
    if not np.all(confidences >= thresholds_per_row - 1e-9):
        raise AssertionError()
    unlabeled_probas = generator.cross_validator.predict_proba(small_unlabeled_data)
    argmax_indices = np.argmax(unlabeled_probas, axis=1)
    max_confidences = unlabeled_probas.max(axis=1)
    expected_thresholds = generator._compute_thresholds_top_k(argmax_indices, max_confidences)
    for cls, expected_threshold in expected_thresholds.items():
        if generator.thresholds_[cls] != pytest.approx(expected_threshold):
            raise AssertionError()


def test_pseudolabelgenerator_fit_fixed_strategy_builds_pseudo_labels_dataframe(
    small_labeled_data, small_unlabeled_data, pseudo_cv_trainer, logistic_regression_estimator
):
    """Tests fit() with the "fixed" threshold strategy produces a valid pseudo-labels frame.

    Args:
        small_labeled_data (tuple): Labeled feature matrix and target vector.
        small_unlabeled_data (pd.DataFrame): Unlabeled feature matrix.
        pseudo_cv_trainer (CrossValidationTrainer): Configured CV trainer.
        logistic_regression_estimator (LogisticRegression): Configured estimator.

    Asserts:
        - Every class threshold equals the configured fixed_threshold
        - Every retained row's confidence is at least the fixed_threshold
    """
    X_train, y_train = small_labeled_data
    generator = PseudoLabelGenerator(
        cross_validator=pseudo_cv_trainer, threshold_method="fixed", fixed_threshold=0.55
    )

    generator.fit(logistic_regression_estimator, X_train, y_train, small_unlabeled_data)

    if not all(t == pytest.approx(0.55) for t in generator.thresholds_.values()):
        raise AssertionError()
    confidences = generator.pseudo_labels_df_["_confidence"].to_numpy()
    if confidences.size and not np.all(confidences >= 0.55 - 1e-9):
        raise AssertionError()


def test_pseudolabelgenerator_fit_auto_strategy_handles_rare_class_fallback(
    imbalanced_labeled_data, small_unlabeled_data, logistic_regression_estimator
):
    """Tests fit() with "auto" completes for a class with fewer than 10 OOF true positives.

    Args:
        imbalanced_labeled_data (tuple): Labeled data where class 1 has 5 samples.
        small_unlabeled_data (pd.DataFrame): Unlabeled feature matrix.
        logistic_regression_estimator (LogisticRegression): Configured estimator.

    Asserts:
        - The minority class indeed has fewer than 10 true-positive OOF samples
        - fit() completes without raising and produces in-range thresholds for every class
    """
    X_train, y_train = imbalanced_labeled_data
    minority_class_count = int((y_train == 1).sum())
    if minority_class_count >= 10:
        raise AssertionError()

    cv = CrossValidationTrainer(
        problem_type="classification",
        metric_name="Accuracy",
        cv=KFold(n_splits=3, shuffle=False),
        verbose=False,
    )
    generator = PseudoLabelGenerator(
        cross_validator=cv, threshold_method="auto", threshold_percentile=80.0
    )

    generator.fit(logistic_regression_estimator, X_train, y_train, small_unlabeled_data)

    if set(generator.thresholds_.keys()) != set(generator.classes_):
        raise AssertionError()
    if not all(0.0 <= t <= 1.0 for t in generator.thresholds_.values()):
        raise AssertionError()


def test_pseudolabelgenerator_compute_thresholds_auto_falls_back_to_predicted_probabilities(
    pseudo_cv_trainer,
):
    """Tests _compute_thresholds_auto falls back to argmax-predicted OOF probabilities.

    Exercised when a class has fewer than 10 true-positive OOF samples but is still
    the argmax prediction for at least one OOF sample.

    Args:
        pseudo_cv_trainer (CrossValidationTrainer): Configured CV trainer.

    Asserts:
        - The computed threshold for the rare class matches a manual recomputation
          based on predicted-probability percentiles from OOF and unlabeled data
    """
    generator = PseudoLabelGenerator(cross_validator=pseudo_cv_trainer, threshold_percentile=80.0)
    generator.classes_ = np.array([0, 1])
    oof_probas = np.array(
        [
            [0.9, 0.1],
            [0.8, 0.2],
            [0.95, 0.05],
            [0.4, 0.6],
            [0.3, 0.7],
            [0.6, 0.4],
        ]
    )
    y_oof = np.array([0, 0, 0, 1, 1, 0])
    generator._unlabeled_probas = np.array(
        [
            [0.7, 0.3],
            [0.2, 0.8],
            [0.5, 0.5],
        ]
    )

    thresholds = generator._compute_thresholds_auto(oof_probas, y_oof)

    pred_mask = np.array([False, False, False, True, True, False])
    unl_cls_mask = np.array([False, True, False])
    expected_t_oof = float(np.percentile(oof_probas[pred_mask, 1], 80.0))
    expected_t_unl = float(np.percentile(generator._unlabeled_probas[unl_cls_mask, 1], 80.0))
    expected_threshold = min(expected_t_oof, expected_t_unl)
    if thresholds[1] != pytest.approx(expected_threshold):
        raise AssertionError()


def test_pseudolabelgenerator_compute_thresholds_auto_uses_degenerate_fallback_when_class_never_predicted(
    pseudo_cv_trainer,
):
    """Tests _compute_thresholds_auto falls back to a neutral 0.5 confidence.

    Exercised when a class has fewer than 10 true-positive OOF samples and is never
    the OOF argmax prediction, so the predicted-probability mask is also empty.

    Args:
        pseudo_cv_trainer (CrossValidationTrainer): Configured CV trainer.

    Asserts:
        - The computed threshold for the never-predicted class equals 0.5
    """
    generator = PseudoLabelGenerator(cross_validator=pseudo_cv_trainer, threshold_percentile=80.0)
    generator.classes_ = np.array([0, 1, 2])
    oof_probas = np.array(
        [
            [0.6, 0.3, 0.1],
            [0.7, 0.2, 0.1],
            [0.55, 0.35, 0.1],
            [0.4, 0.5, 0.1],
        ]
    )
    y_oof = np.array([0, 0, 1, 1])
    generator._unlabeled_probas = np.array(
        [
            [0.5, 0.4, 0.1],
            [0.6, 0.3, 0.1],
        ]
    )

    thresholds = generator._compute_thresholds_auto(oof_probas, y_oof)

    if thresholds[2] != pytest.approx(0.5):
        raise AssertionError()


def test_pseudolabelgenerator_compute_thresholds_top_k_sets_threshold_to_one_when_no_candidates(
    pseudo_cv_trainer,
):
    """Tests _compute_thresholds_top_k selects nothing for a class with no argmax candidates.

    Args:
        pseudo_cv_trainer (CrossValidationTrainer): Configured CV trainer.

    Asserts:
        - The threshold for the class with zero candidates equals 1.0
        - The threshold for the class with candidates matches a manual percentile
    """
    generator = PseudoLabelGenerator(
        cross_validator=pseudo_cv_trainer, threshold_method="top_k", top_k_pct=0.5
    )
    generator.classes_ = np.array([0, 1])
    argmax_indices = np.zeros(6, dtype=int)
    max_confidences = np.array([0.9, 0.8, 0.7, 0.6, 0.55, 0.51])

    thresholds = generator._compute_thresholds_top_k(argmax_indices, max_confidences)

    if thresholds[1] != pytest.approx(1.0):
        raise AssertionError()
    expected_threshold_0 = float(np.percentile(max_confidences, 50.0))
    if thresholds[0] != pytest.approx(expected_threshold_0):
        raise AssertionError()


def test_pseudolabelgenerator_density_curve_uses_kde_for_sufficient_varied_samples():
    """Tests _density_curve uses a Gaussian KDE when enough varied samples are given.

    Asserts:
        - The returned density array matches the grid shape
        - All density values are finite and non-negative
    """
    rng = np.random.default_rng(3)
    values = rng.normal(loc=0.7, scale=0.05, size=50)
    grid = np.linspace(0.0, 1.0, 100)

    density = PseudoLabelGenerator._density_curve(values, grid)

    if density.shape != grid.shape:
        raise AssertionError()
    if not np.all(np.isfinite(density)):
        raise AssertionError()
    if not np.all(density >= 0.0):
        raise AssertionError()


@pytest.mark.parametrize(
    "values",
    [
        np.array([0.8, 0.81, 0.79]),
        np.array([0.5, 0.5, 0.5, 0.5, 0.5, 0.5]),
    ],
)
def test_pseudolabelgenerator_density_curve_falls_back_to_histogram_for_small_or_constant_samples(
    values,
):
    """Tests _density_curve falls back to an interpolated histogram in unstable cases.

    Exercised when the sample has fewer than 5 points, or is effectively constant
    (std close to zero), both of which make kernel density estimation unstable.

    Args:
        values (np.ndarray): Small or near-constant confidence sample under test.

    Asserts:
        - The returned density array matches the grid shape
        - All density values are finite and non-negative
    """
    grid = np.linspace(0.0, 1.0, 100)

    density = PseudoLabelGenerator._density_curve(values, grid)

    if density.shape != grid.shape:
        raise AssertionError()
    if not np.all(np.isfinite(density)):
        raise AssertionError()
    if not np.all(density >= 0.0):
        raise AssertionError()


def test_pseudolabelgenerator_plot_pseudo_label_stats_raises_before_fit(pseudo_cv_trainer):
    """Tests plot_pseudo_label_stats raises before fit() has been called.

    Args:
        pseudo_cv_trainer (CrossValidationTrainer): Configured CV trainer.

    Asserts:
        - RuntimeError is raised instructing the caller to fit first
    """
    generator = PseudoLabelGenerator(cross_validator=pseudo_cv_trainer)

    with pytest.raises(
        RuntimeError, match=r"Call fit\(\) before plot_pseudo_label_stats\(\)\."
    ):
        generator.plot_pseudo_label_stats()


def test_pseudolabelgenerator_plot_pseudo_label_stats_runs_without_raising_on_normal_data(
    small_labeled_data, small_unlabeled_data, pseudo_cv_trainer, logistic_regression_estimator
):
    """Tests plot_pseudo_label_stats completes for a normally sized unlabeled set.

    Args:
        small_labeled_data (tuple): Labeled feature matrix and target vector.
        small_unlabeled_data (pd.DataFrame): Unlabeled feature matrix.
        pseudo_cv_trainer (CrossValidationTrainer): Configured CV trainer.
        logistic_regression_estimator (LogisticRegression): Configured estimator.

    Asserts:
        - The method completes without raising any exception
    """
    X_train, y_train = small_labeled_data
    generator = PseudoLabelGenerator(
        cross_validator=pseudo_cv_trainer, threshold_method="fixed", fixed_threshold=0.3
    )
    generator.fit(logistic_regression_estimator, X_train, y_train, small_unlabeled_data)

    generator.plot_pseudo_label_stats()


def test_pseudolabelgenerator_plot_pseudo_label_stats_runs_without_raising_on_tiny_unlabeled_set(
    small_labeled_data, pseudo_cv_trainer, logistic_regression_estimator
):
    """Tests plot_pseudo_label_stats completes for a tiny unlabeled set.

    A tiny candidate pool per class exercises the histogram-fallback branch of
    _density_curve inside the full plotting pipeline.

    Args:
        small_labeled_data (tuple): Labeled feature matrix and target vector.
        pseudo_cv_trainer (CrossValidationTrainer): Configured CV trainer.
        logistic_regression_estimator (LogisticRegression): Configured estimator.

    Asserts:
        - The method completes without raising any exception
    """
    X_train, y_train = small_labeled_data
    tiny_unlabeled = pd.DataFrame(
        np.random.default_rng(1).normal(size=(3, _N_FEATURES)), columns=_FEATURE_NAMES
    )
    generator = PseudoLabelGenerator(
        cross_validator=pseudo_cv_trainer, threshold_method="fixed", fixed_threshold=0.3
    )
    generator.fit(logistic_regression_estimator, X_train, y_train, tiny_unlabeled)

    generator.plot_pseudo_label_stats()


def test_pseudolabelgenerator_get_augmented_dataset_raises_before_fit(pseudo_cv_trainer):
    """Tests get_augmented_dataset raises before fit() has been called.

    Args:
        pseudo_cv_trainer (CrossValidationTrainer): Configured CV trainer.

    Asserts:
        - RuntimeError is raised instructing the caller to fit first
    """
    generator = PseudoLabelGenerator(cross_validator=pseudo_cv_trainer)

    with pytest.raises(
        RuntimeError, match=r"Call fit\(\) before get_augmented_dataset\(\)\."
    ):
        generator.get_augmented_dataset()


def test_pseudolabelgenerator_get_augmented_dataset_weights_rows_by_confidence(
    small_labeled_data, small_unlabeled_data, pseudo_cv_trainer, logistic_regression_estimator
):
    """Tests get_augmented_dataset assigns weight 1.0 to originals and confidence to pseudo rows.

    Args:
        small_labeled_data (tuple): Labeled feature matrix and target vector.
        small_unlabeled_data (pd.DataFrame): Unlabeled feature matrix.
        pseudo_cv_trainer (CrossValidationTrainer): Configured CV trainer.
        logistic_regression_estimator (LogisticRegression): Configured estimator.

    Asserts:
        - At least one pseudo-labeled row was selected
        - Combined dataset length equals original plus pseudo-labeled row counts
        - Original rows all receive sample weight 1.0
        - Pseudo-labeled rows receive weight equal to their predicted confidence
    """
    X_train, y_train = small_labeled_data
    generator = PseudoLabelGenerator(
        cross_validator=pseudo_cv_trainer, threshold_method="fixed", fixed_threshold=0.3
    )
    generator.fit(logistic_regression_estimator, X_train, y_train, small_unlabeled_data)

    x_combined, y_combined, w_combined = generator.get_augmented_dataset()

    n_original = len(X_train)
    n_pseudo = len(generator.pseudo_labels_df_)
    if n_pseudo == 0:
        raise AssertionError()
    if len(x_combined) != n_original + n_pseudo:
        raise AssertionError()
    if len(y_combined) != n_original + n_pseudo:
        raise AssertionError()
    if not np.allclose(w_combined.iloc[:n_original].to_numpy(), 1.0):
        raise AssertionError()
    expected_pseudo_weights = generator.pseudo_labels_df_["_confidence"].to_numpy()
    if not np.allclose(w_combined.iloc[n_original:].to_numpy(), expected_pseudo_weights):
        raise AssertionError()


if __name__ == "__main__":
    print("Run this file with pytest to execute tests.")
