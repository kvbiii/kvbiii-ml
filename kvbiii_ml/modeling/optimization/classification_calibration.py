from __future__ import annotations

import sys
from copy import deepcopy
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.calibration import calibration_curve
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import BaseCrossValidator

sys.path.append(str(Path(__file__).resolve().parents[3]))
from kvbiii_ml.evaluation.metrics import (
    METRICS,
    get_metric_direction,
    get_metric_function,
    get_metric_type,
)
from kvbiii_ml.modeling.training.base_trainer import BaseTrainer
from kvbiii_ml.modeling.training.calibrated_model import CalibratedModel
from kvbiii_ml.modeling.training.cross_validation import CrossValidationTrainer
from kvbiii_ml.modeling.training.oof_model import (
    OOFModel,
    _apply_calibrator,
    _collect_oof_probas,
    _disable_early_stopping,
    _fit_calibrator,
    _get_n_iterations,
    _set_n_iterations,
)

_DEFAULT_SEED = 17
_DEFAULT_N_BINS = 10
_DEFAULT_METHODS = ("isotonic", "sigmoid")
_BUILTIN_SELECTION_COLS: dict[str, tuple[str, str]] = {
    "ECE": ("ECE", "minimize"),
    "Brier": ("Brier", "minimize"),
    "Brier Score": ("Brier", "minimize"),
}


def _compute_ece(
    y_true: np.ndarray,
    probas: np.ndarray,
    n_bins: int,
    classes: np.ndarray,
) -> float:
    """Compute Expected Calibration Error (ECE).

    Binary: ECE on the positive-class probabilities.
    Multiclass: average ECE across classes using OVR.

    Args:
        y_true (np.ndarray): True labels of shape (n_samples,).
        probas (np.ndarray): Predicted probabilities of shape (n_samples, n_classes).
        n_bins (int): Number of equally-spaced bins in [0, 1].
        classes (np.ndarray): Unique class labels.

    Returns:
        float: Expected Calibration Error in [0, 1].
    """
    n_samples = len(y_true)

    def _ece_binary(y_bin: np.ndarray, p: np.ndarray) -> float:
        bin_boundaries = np.linspace(0.0, 1.0, n_bins + 1)
        bin_ids = np.digitize(p, bin_boundaries[1:-1])
        return float(
            sum(
                (mask.sum() / n_samples) * abs(y_bin[mask].mean() - p[mask].mean())
                for b in range(n_bins)
                if (mask := bin_ids == b).any()
            )
        )

    if len(classes) == 2:
        return _ece_binary((y_true == classes[1]).astype(int), probas[:, 1])
    return float(
        np.mean(
            [
                _ece_binary((y_true == classes[i]).astype(int), probas[:, i])
                for i in range(len(classes))
            ]
        )
    )


def _eval_calibrator_oof(
    oof_probas: np.ndarray,
    y_oof: np.ndarray,
    method: str,
    classes: np.ndarray,
    cv_splitter: BaseCrossValidator,
) -> np.ndarray:
    """Evaluate a calibrator honestly via nested CV on OOF probabilities.

    Fits the calibrator on k-1 folds of the OOF data and scores on the held-out
    fold, repeating for every fold. This avoids the in-sample optimism that occurs
    when fitting and evaluating a calibrator on the same OOF predictions.

    Args:
        oof_probas (np.ndarray): OOF probabilities of shape (n_samples, n_classes).
        y_oof (np.ndarray): OOF true labels of shape (n_samples,).
        method (str): Calibration method, either "isotonic" or "sigmoid".
        classes (np.ndarray): Unique class labels.
        cv_splitter (BaseCrossValidator): Splitter for the inner calibration CV.

    Returns:
        np.ndarray: Calibrated probabilities of shape (n_samples, n_classes) where
            each row was produced by a calibrator that never saw that row during fit.
    """
    cal_probas = np.zeros_like(oof_probas)
    for train_idx, val_idx in cv_splitter.split(oof_probas, y_oof):
        cal = _fit_calibrator(oof_probas[train_idx], y_oof[train_idx], method, classes)
        cal_probas[val_idx] = _apply_calibrator(
            cal, oof_probas[val_idx], classes, method
        )
    return cal_probas


class ClassificationCalibrator:
    """Select and apply the best post-hoc calibration strategy via OOF probabilities.

    Runs cross-validation once with the base estimator to collect out-of-fold
    probabilities, evaluates uncalibrated and calibrated variants using the
    cross-validator's metric, then exposes the best strategy as ``best_estimator_``
    for use in a subsequent CrossValidationTrainer.fit() call.

    ``best_estimator_`` is either a CalibratedModel (when calibration wins) or the
    original estimator (when uncalibrated wins). CalibratedModel calibrates via a
    single hold-out split rather than an inner cross-validation, so re-using it inside
    another CrossValidationTrainer.fit() stays linear in the number of outer folds
    instead of triggering nested cross-validation.

    Supports single estimators (including CatBoost with categorical features),
    EnsembleModel instances (treated as black-box), and preprocessing pipelines
    defined in the cross-validator.
    """

    def __init__(
        self,
        estimator: BaseEstimator,
        cross_validator: CrossValidationTrainer,
        methods: Iterable[str] | None = None,
        n_bins: int = _DEFAULT_N_BINS,
        seed: int = _DEFAULT_SEED,
        id2label: dict[int, str] | None = None,
        selection_metric_name: str | None = "Brier Score",
        calibration_holdout_size: float = 0.2,
    ) -> None:
        """Initialize the calibrator.

        Args:
            estimator (BaseEstimator): Unfitted estimator or EnsembleModel to calibrate.
            cross_validator (CrossValidationTrainer): Configured CV trainer. Its cv
                splitter is inherited by the inner OOFModel for calibration loops.
            methods (Iterable[str] | None): Calibration methods to evaluate. Accepts
                "isotonic" and/or "sigmoid". Defaults to ("isotonic", "sigmoid").
            n_bins (int): Number of bins for ECE computation and calibration curves.
                Defaults to 10.
            seed (int): Random seed (reserved for future use). Defaults to 17.
            id2label (dict[int, str] | None): Optional class-index to label mapping
                for plot titles. Defaults to None.
            selection_metric_name (str | None): Metric used to choose the best
                calibration method. Accepts "ECE", "Brier Score" (both always computed
                internally), None (falls back to the cross_validator's metric), or any
                key from METRICS (computed on OOF probabilities and added as an extra
                column). Defaults to "Brier Score".
            calibration_holdout_size (float): Fraction of the training data the winning
                CalibratedModel holds out to fit its calibrator. Defaults to 0.2.

        Raises:
            ValueError: If selection_metric_name is not None, a built-in name, or a
                valid METRICS key.
        """
        if (
            selection_metric_name is not None
            and selection_metric_name not in _BUILTIN_SELECTION_COLS
            and selection_metric_name not in METRICS
        ):
            raise ValueError(
                f"selection_metric_name '{selection_metric_name}' is not recognised. "
                f"Use None, 'ECE', 'Brier Score', or any key from METRICS."
            )
        self.estimator = estimator
        self.cross_validator = cross_validator
        self.methods = tuple(methods or _DEFAULT_METHODS)
        self.n_bins = n_bins
        self.seed = seed
        self.id2label = id2label
        self.selection_metric_name = selection_metric_name
        self.calibration_holdout_size = calibration_holdout_size

        self.train_scores_: list[float] = []
        self.valid_scores_: list[float] = []
        self.calibrated_valid_scores_: list[float] = []
        self.calibration_scores_df_: pd.DataFrame = pd.DataFrame()
        self.best_method_: str = ""
        self.best_calibrator_: IsotonicRegression | LogisticRegression | list | None = (
            None
        )
        self.best_estimator_: CalibratedModel | BaseEstimator | None = None

        self._oof_probas_by_method: dict[str, np.ndarray] = {}
        self._y_oof: np.ndarray | None = None
        self._classes: np.ndarray | None = None
        self._fitted_full_estimator: BaseEstimator | None = None
        self._fitted_full_pipeline = None

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        eval_set: list[tuple[pd.DataFrame, pd.Series]] | None = None,
    ) -> "ClassificationCalibrator":
        """Run CV calibration, evaluate methods, and store the best strategy.

        Args:
            X (pd.DataFrame): Full training feature matrix.
            y (pd.Series): Full training target vector.
            eval_set (list[tuple[pd.DataFrame, pd.Series]] | None): Validation data for
                early stopping during the final full-data refit. Accepts the same format
                as EnsembleModel: [(X_train, y_train), (X_valid, y_valid)] or
                [(X_valid, y_valid)]. X values are expected raw (unprocessed); the
                pipeline from cross_validator is applied automatically. Defaults to None.

        Returns:
            ClassificationCalibrator: Fitted self. After fitting, ``calibrated_valid_scores_``
                holds per-fold scores for the winning method computed directly from the stored
                OOF probabilities — no additional model fitting required, so they remain the
                cheapest honest estimate of calibrated CV performance. ``best_estimator_`` is a
                CalibratedModel suitable for re-use in a downstream ``CrossValidationTrainer.fit``
                (e.g. pseudo-labeling experiments); it calibrates via a single hold-out split and
                does not trigger nested cross-validation.
        """
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)
        if not isinstance(y, pd.Series):
            y = pd.Series(np.asarray(y).ravel())

        train_scores, valid_scores, _ = self.cross_validator.fit(self.estimator, X, y)
        self.train_scores_ = list(train_scores)
        self.valid_scores_ = list(valid_scores)

        y_oof, oof_probas = _collect_oof_probas(self.cross_validator, X, y)
        self._y_oof = y_oof
        self._classes = np.unique(y_oof)

        metric_fn = self.cross_validator.metric_fn
        metric_type = self.cross_validator.metric_type
        metric_direction = self.cross_validator.metric_direction

        uncal_preds = self._probas_to_metric_input(oof_probas, metric_type)
        uncal_score = float(metric_fn(y_oof, uncal_preds))
        rows: list[dict] = [
            {
                "Method": "Uncalibrated",
                self.cross_validator.metric_name: uncal_score,
                "ECE": _compute_ece(y_oof, oof_probas, self.n_bins, self._classes),
                "Brier": self._compute_brier(y_oof, oof_probas, self._classes),
            }
        ]
        self._oof_probas_by_method["Uncalibrated"] = oof_probas

        calibrators: dict[str, IsotonicRegression | LogisticRegression | list] = {}
        cal_cv = self.cross_validator.cv

        for method in self.methods:
            eval_probas = _eval_calibrator_oof(
                oof_probas, y_oof, method, self._classes, cal_cv
            )
            cal = _fit_calibrator(oof_probas, y_oof, method, self._classes)
            calibrators[method] = cal
            cal_preds = self._probas_to_metric_input(eval_probas, metric_type)
            rows.append(
                {
                    "Method": method.capitalize(),
                    self.cross_validator.metric_name: float(
                        metric_fn(y_oof, cal_preds)
                    ),
                    "ECE": _compute_ece(y_oof, eval_probas, self.n_bins, self._classes),
                    "Brier": self._compute_brier(y_oof, eval_probas, self._classes),
                }
            )
            self._oof_probas_by_method[method.capitalize()] = eval_probas

        self.calibration_scores_df_ = pd.DataFrame(rows).set_index("Method")

        sel_name = self.selection_metric_name
        if sel_name is None or sel_name == self.cross_validator.metric_name:
            sel_col = self.cross_validator.metric_name
            sel_direction = metric_direction
        elif sel_name in _BUILTIN_SELECTION_COLS:
            sel_col, sel_direction = _BUILTIN_SELECTION_COLS[sel_name]
        else:
            sel_fn = get_metric_function(sel_name)
            sel_direction = get_metric_direction(sel_name)
            sel_type = get_metric_type(sel_name)
            self.calibration_scores_df_[sel_name] = pd.Series(
                {
                    method_key: float(
                        sel_fn(
                            y_oof,
                            self._probas_to_metric_input(probas, sel_type),
                        )
                    )
                    for method_key, probas in self._oof_probas_by_method.items()
                }
            )
            sel_col = sel_name

        best_row = (
            self.calibration_scores_df_[sel_col].idxmax()
            if sel_direction == "maximize"
            else self.calibration_scores_df_[sel_col].idxmin()
        )
        self.best_method_ = str(best_row)

        best_probas = self._oof_probas_by_method[str(best_row)]
        y_arr = np.asarray(y)
        self.calibrated_valid_scores_ = [
            float(
                metric_fn(
                    y_arr[val_idx],
                    self._probas_to_metric_input(best_probas[val_idx], metric_type),
                )
            )
            for _, val_idx in self.cross_validator.cv.split(X, y)
        ]

        if best_row == "Uncalibrated":
            best_cal_method = self.methods[0]
            self.best_calibrator_ = calibrators[best_cal_method]
        else:
            best_cal_method = best_row.lower()
            self.best_calibrator_ = calibrators[best_cal_method]

        if best_row != "Uncalibrated":
            self.best_estimator_ = CalibratedModel(
                estimator=deepcopy(self.estimator),
                problem_type="classification",
                calibration_method=best_cal_method,
                calibration_size=self.calibration_holdout_size,
                seed=self.seed,
            )
        else:
            self.best_estimator_ = self.estimator

        self._refit_full_estimator(X, y, eval_set)
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Return calibrated class probabilities using the full-data-fitted estimator.

        Args:
            X (pd.DataFrame): Features to predict (raw, unprocessed).

        Returns:
            np.ndarray: Calibrated probabilities of shape (n_samples, n_classes).

        Raises:
            RuntimeError: If called before fit().
        """
        if self._fitted_full_estimator is None:
            raise RuntimeError("Call fit() before predict_proba().")

        X_proc = CrossValidationTrainer._transform_with_pipeline(
            self._fitted_full_pipeline, X
        )
        X_ord = CrossValidationTrainer._order_X_for_estimator(
            X_proc, self._fitted_full_estimator
        )
        raw = self._fitted_full_estimator.predict_proba(X_ord)
        if raw.ndim == 1:
            raw = np.column_stack([1 - raw, raw])
        best_method_key = (
            self.best_method_.lower()
            if self.best_method_ != "Uncalibrated"
            else self.methods[0]
        )
        return _apply_calibrator(
            self.best_calibrator_, raw, self._classes, best_method_key
        )

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Return class predictions using the calibrated full estimator.

        Args:
            X (pd.DataFrame): Features to predict (raw, unprocessed).

        Returns:
            np.ndarray: Predicted class labels of shape (n_samples,).
        """
        return self._classes[np.argmax(self.predict_proba(X), axis=1)]

    def plot_calibration_curves(self) -> None:
        """Plot reliability diagrams for each calibration method.

        Binary: one subplot per method in a single row.
        Multiclass: grid of (methods x classes) subplots.

        Raises:
            RuntimeError: If called before fit().
        """
        if not self._oof_probas_by_method:
            raise RuntimeError("Call fit() before plot_calibration_curves().")

        y_true = self._y_oof
        is_binary = len(self._classes) == 2
        method_names = list(self._oof_probas_by_method.keys())
        n_methods = len(method_names)

        if is_binary:
            fig, axes = plt.subplots(
                1, n_methods, figsize=(5 * n_methods, 5), squeeze=False
            )
            for col, name in enumerate(method_names):
                ax = axes[0, col]
                probas = self._oof_probas_by_method[name]
                frac_pos, mean_pred = calibration_curve(
                    y_true, probas[:, 1], n_bins=self.n_bins
                )
                ax.plot(mean_pred, frac_pos, marker="o", label=name)
                ax.plot([0, 1], [0, 1], "k:", label="Perfect")
                ax.set_xlabel("Mean predicted probability")
                ax.set_ylabel("Fraction of positives")
                ax.set_title(name + (" ★" if name == self.best_method_ else ""))
                ax.legend()
            fig.suptitle("Calibration Curves (Binary)", fontsize=13, fontweight="bold")
            plt.tight_layout()
            plt.show()
        else:
            n_classes = len(self._classes)
            labels = self._resolve_labels(n_classes, self.id2label)
            fig, axes = plt.subplots(
                n_methods,
                n_classes,
                figsize=(4 * n_classes, 4 * n_methods),
                squeeze=False,
            )
            for row, name in enumerate(method_names):
                probas = self._oof_probas_by_method[name]
                for col, cls in enumerate(self._classes):
                    ax = axes[row, col]
                    y_bin = (y_true == cls).astype(int)
                    frac_pos, mean_pred = calibration_curve(
                        y_bin, probas[:, col], n_bins=self.n_bins
                    )
                    ax.plot(mean_pred, frac_pos, marker="o")
                    ax.plot([0, 1], [0, 1], "k:")
                    ax.set_xlabel("Mean predicted probability")
                    if col == 0:
                        ax.set_ylabel(
                            name if name != self.best_method_ else f"{name} ★",
                            fontsize=11,
                            fontweight="bold",
                        )
                    ax.set_title(f"Class {labels[col]}")
            fig.suptitle(
                "Calibration Curves (Multiclass)", fontsize=13, fontweight="bold"
            )
            plt.tight_layout()
            plt.show()

    def _refit_full_estimator(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        eval_set: list[tuple[pd.DataFrame, pd.Series]] | None = None,
    ) -> None:
        """Refit the base estimator on full training data for predict_proba().

        Applies the cross-validator's preprocessing pipeline (clone + fit_transform),
        then refits the estimator using the average iteration count from OOF folds.
        When eval_set is provided, early stopping is preserved for the refit.

        Args:
            X (pd.DataFrame): Full raw feature matrix.
            y (pd.Series): Full training target.
            eval_set (list[tuple] | None): Validation data for early stopping. X values
                are raw and are transformed through the pipeline automatically. Defaults to None.
        """
        from sklearn.base import clone as sklearn_clone

        pipeline = self.cross_validator.preprocessing_pipeline
        fitted_pipeline = sklearn_clone(pipeline) if pipeline is not None else None
        X_proc = (
            fitted_pipeline.fit_transform(X, y) if fitted_pipeline is not None else X
        )
        self._fitted_full_pipeline = fitted_pipeline

        X_valid, y_valid = OOFModel._parse_eval_set(eval_set)
        if X_valid is not None and fitted_pipeline is not None:
            X_valid = fitted_pipeline.transform(X_valid)

        n_iter_list = [
            n
            for est in self.cross_validator.fitted_estimators_
            if (n := _get_n_iterations(est)) is not None
        ]
        final_est = deepcopy(self.estimator)
        if n_iter_list:
            _set_n_iterations(final_est, int(np.mean(n_iter_list)))
        if X_valid is None:
            _disable_early_stopping(final_est)
        self._fitted_full_estimator = BaseTrainer.fit_estimator(
            final_est, X_proc, y, X_valid, y_valid
        )

    @staticmethod
    def _probas_to_metric_input(probas: np.ndarray, metric_type: str) -> np.ndarray:
        """Convert probability matrix to the format expected by the metric function.

        Args:
            probas (np.ndarray): Probabilities of shape (n_samples, n_classes).
            metric_type (str): Either "probs" or "preds".

        Returns:
            np.ndarray: Positive-class column for binary "probs", argmax predictions
                for "preds", or full matrix for multiclass "probs".
        """
        if metric_type == "probs":
            return probas[:, 1] if probas.shape[1] == 2 else probas
        return np.argmax(probas, axis=1)

    @staticmethod
    def _compute_brier(
        y_true: np.ndarray, probas: np.ndarray, classes: np.ndarray
    ) -> float:
        """Compute average Brier score across classes (OVR).

        Args:
            y_true (np.ndarray): True labels of shape (n_samples,).
            probas (np.ndarray): Predicted probabilities of shape (n_samples, n_classes).
            classes (np.ndarray): Unique class labels.

        Returns:
            float: Mean Brier score.
        """
        return float(
            np.mean(
                [
                    float(np.mean((probas[:, i] - (y_true == cls).astype(float)) ** 2))
                    for i, cls in enumerate(classes)
                ]
            )
        )

    @staticmethod
    def _resolve_labels(n_classes: int, id2label: dict[int, str] | None) -> list[str]:
        """Map class indices to display labels.

        Args:
            n_classes (int): Number of classes.
            id2label (dict[int, str] | None): Optional mapping.

        Returns:
            list[str]: Display labels for each class.
        """
        if id2label is None:
            return [str(i) for i in range(n_classes)]
        return [id2label.get(i, str(i)) for i in range(n_classes)]


if __name__ == "__main__":
    import numpy as np
    import pandas as pd
    from lightgbm import LGBMClassifier
    from xgboost import XGBClassifier
    from catboost import CatBoostClassifier
    from sklearn.datasets import make_classification
    from sklearn.model_selection import StratifiedKFold
    from sklearn.pipeline import Pipeline

    from kvbiii_ml.data_processing.preprocessing.outlier_handling.winsorizer_trimmer import (
        WinsorizerWithOriginal,
    )
    from kvbiii_ml.modeling.training.ensemble_model import EnsembleModel

    RANDOM_STATE = 42
    N_SAMPLES = 3_000
    N_FEATURES = 10
    N_FOLDS = 3
    ES = 30
    FEATURE_NAMES = [f"feature_{i}" for i in range(N_FEATURES)]

    def _make_clf_data(n_classes: int = 2) -> tuple[pd.DataFrame, pd.Series]:
        """Generate synthetic classification dataset."""
        X_arr, y_arr = make_classification(
            n_samples=N_SAMPLES,
            n_features=N_FEATURES,
            n_informative=6,
            n_redundant=2,
            n_classes=n_classes,
            n_clusters_per_class=1,
            random_state=RANDOM_STATE,
        )
        return pd.DataFrame(X_arr, columns=FEATURE_NAMES), pd.Series(
            y_arr, name="target"
        )

    def _build_pipeline() -> Pipeline:
        """Build a preprocessing pipeline with outlier handling."""
        return Pipeline(
            [
                (
                    "winsorizer",
                    WinsorizerWithOriginal(
                        variables=FEATURE_NAMES,
                        capping_method="gaussian",
                        tail="both",
                    ),
                ),
            ]
        )

    def _build_cv(
        metric: str, pipeline: Pipeline | None = None
    ) -> CrossValidationTrainer:
        """Build a CrossValidationTrainer for the given metric."""
        return CrossValidationTrainer(
            metric_name=metric,
            problem_type="classification",
            cv=StratifiedKFold(
                n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE
            ),
            preprocessing_pipeline=pipeline,
            verbose=False,
        )

    print("=" * 75)
    print("Scenario 1: Binary + LightGBM + preprocessing pipeline")
    print("=" * 75)
    X_bin, y_bin = _make_clf_data(n_classes=2)
    lgbm_bin = LGBMClassifier(
        n_estimators=200,
        early_stopping_rounds=ES,
        verbose=-1,
        random_state=RANDOM_STATE,
    )
    cv_1 = _build_cv("Log Loss", _build_pipeline())
    cal_1 = ClassificationCalibrator(estimator=lgbm_bin, cross_validator=cv_1)
    cal_1.fit(X_bin, y_bin)
    print(f"Best method: {cal_1.best_method_}")
    print(cal_1.calibration_scores_df_)
    print("Passing best_estimator_ to cross_validator.fit():")
    _, valid_s, _ = cv_1.fit(cal_1.best_estimator_, X_bin, y_bin)
    print(f"  Calibrated CV valid: {np.mean(valid_s):.4f} ± {np.std(valid_s):.4f}")
    cal_1.plot_calibration_curves()

    print("\n" + "=" * 75)
    print("Scenario 2: Binary + CatBoost with categorical dtype columns")
    print("=" * 75)
    try:
        rng = np.random.RandomState(RANDOM_STATE)
        X_cat = X_bin.copy()
        X_cat["cat_feat"] = pd.Categorical(rng.choice(["A", "B", "C"], size=N_SAMPLES))
        cat_cb = CatBoostClassifier(
            iterations=200,
            early_stopping_rounds=ES,
            verbose=0,
            random_state=RANDOM_STATE,
        )
        cv_2 = _build_cv("Log Loss")
        for label, est, X_in in [
            ("LightGBM", lgbm_bin, X_bin),
            ("CatBoost", cat_cb, X_cat),
        ]:
            cal_2 = ClassificationCalibrator(estimator=est, cross_validator=cv_2)
            cal_2.fit(X_in, y_bin)
            print(f"  {label} - best: {cal_2.best_method_}")
            print(cal_2.calibration_scores_df_)
    except ImportError:
        print("  CatBoost not installed; skipping CatBoost sub-scenario.")

    print("\n" + "=" * 75)
    print("Scenario 3: EnsembleModel - black-box calibration")
    print("=" * 75)
    estimators = [
        LGBMClassifier(
            n_estimators=200,
            early_stopping_rounds=ES,
            verbose=-1,
            random_state=RANDOM_STATE,
        ),
        # XGBClassifier(
        #     n_estimators=200,
        #     early_stopping_rounds=ES,
        #     verbosity=0,
        #     random_state=RANDOM_STATE + 1,
        # ),
        CatBoostClassifier(
            iterations=200,
            early_stopping_rounds=ES,
            verbose=0,
            random_state=RANDOM_STATE + 2,
        ),
        CatBoostClassifier(
            iterations=200,
            early_stopping_rounds=ES,
            verbose=0,
            random_state=RANDOM_STATE + 2,
        ),
        CatBoostClassifier(
            iterations=200,
            early_stopping_rounds=ES,
            verbose=0,
            random_state=RANDOM_STATE + 2,
        ),
        CatBoostClassifier(
            iterations=200,
            early_stopping_rounds=ES,
            verbose=0,
            random_state=RANDOM_STATE + 2,
        ),
    ]
    ensemble = EnsembleModel(estimators=estimators, problem_type="classification")
    cv_3 = _build_cv("Log Loss")
    cal_3 = ClassificationCalibrator(
        estimator=ensemble, cross_validator=cv_3, selection_metric_name="ECE"
    )
    cal_3.fit(X_bin, y_bin)
    print(f"  best_method_: {cal_3.best_method_}")
    print(
        f"  best_estimator_ is CalibratedModel: {isinstance(cal_3.best_estimator_, CalibratedModel)}"
    )
    print(cal_3.calibration_scores_df_)
    import time

    start = time.perf_counter()
    _, valid_s_3, _ = cv_3.fit(cal_3.best_estimator_, X_bin, y_bin)
    print(
        f"  Time for CV fit with best_estimator_: {time.perf_counter() - start:.2f} seconds"
    )
    print(f"  Calibrated CV valid: {np.mean(valid_s_3):.4f} ± {np.std(valid_s_3):.4f}")
    cal_3.plot_calibration_curves()

    print("\n" + "=" * 75)
    print(
        "Scenario 4: Multiclass + LightGBM + eval_set for early stopping on final refit"
    )
    print("=" * 75)
    X_mc, y_mc = _make_clf_data(n_classes=3)
    lgbm_mc = LGBMClassifier(
        n_estimators=200,
        early_stopping_rounds=ES,
        verbose=-1,
        random_state=RANDOM_STATE,
    )
    cv_4 = _build_cv("Brier Score", _build_pipeline())
    cal_4 = ClassificationCalibrator(estimator=lgbm_mc, cross_validator=cv_4)
    from sklearn.model_selection import train_test_split

    X_tr, X_val, y_tr, y_val = train_test_split(
        X_mc, y_mc, test_size=0.2, random_state=RANDOM_STATE, stratify=y_mc
    )
    cal_4.fit(X_tr, y_tr, eval_set=[(X_val, y_val)])
    print(f"  best_method_: {cal_4.best_method_}")
    print(cal_4.calibration_scores_df_)
    print("Passing best_estimator_ to cross_validator.fit():")
    _, valid_s_mc, _ = cv_4.fit(cal_4.best_estimator_, X_mc, y_mc)
    print(
        f"  Calibrated CV valid: {np.mean(valid_s_mc):.4f} ± {np.std(valid_s_mc):.4f}"
    )
    cal_4.plot_calibration_curves()
