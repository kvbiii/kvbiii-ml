from __future__ import annotations

import io
import sys
from collections.abc import Iterable
from copy import deepcopy
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from pandas.io.formats.style import Styler
from sklearn.base import BaseEstimator, clone
from sklearn.calibration import calibration_curve
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression

sys.path.append(str(Path(__file__).resolve().parents[3]))
from kvbiii_ml.evaluation.generate_reports import _apply_fancy_styling
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

try:  # optional notebook dependency
    from IPython import get_ipython
    from IPython.display import Image, display

    def _show_figure(fig: Figure) -> None:
        """Display fig inline in a notebook without requiring `%matplotlib inline`.

        Rasterizes fig to PNG bytes and hands them to IPython's display system
        directly, rather than calling `plt.show()`, which only renders inline
        when the active matplotlib backend is already the notebook's inline
        backend. Outside a notebook kernel (plain script), falls back to
        `plt.show()`.

        Args:
            fig (Figure): Figure to display.
        """
        if get_ipython() is None:
            plt.show()
            return
        buf = io.BytesIO()
        fig.savefig(
            buf,
            format="png",
            dpi=fig.dpi,
            facecolor=fig.get_facecolor(),
            bbox_inches="tight",
        )
        display(Image(data=buf.getvalue()))
        plt.close(fig)

except ImportError:  # pragma: no cover

    def _show_figure(fig: Figure) -> None:
        """Fallback for environments without IPython installed.

        Args:
            fig (Figure): Figure to display.
        """
        plt.show()


_DEFAULT_SEED = 17
_DEFAULT_N_BINS = 10
_DEFAULT_METHODS = ("isotonic", "sigmoid")
_UNCALIBRATED = "Uncalibrated"
_BUILTIN_SELECTION_COLS: dict[str, tuple[str, str]] = {
    "ECE": ("ECE", "minimize"),
    "Brier": ("Brier", "minimize"),
    "Brier Score": ("Brier", "minimize"),
}
_brier_score = get_metric_function("Brier Score")

_PLOT_SURFACE = "#fcfcfb"
_PLOT_ACCENT = "#2a78d6"
_PLOT_ACCENT_TINT = "#b7d3f6"
_PLOT_REFERENCE = "#898781"
_PLOT_GRID = "#e1e0d9"
_PLOT_INK = "#0b0b0b"
_PLOT_MUTED_INK = "#52514e"
_PLOT_WIN = "#0ca30c"


def _binary_ece(y_bin: np.ndarray, p: np.ndarray, n_bins: int) -> float:
    """Compute Expected Calibration Error for one binary indicator/probability pair.

    Vectorised via ``np.bincount`` rather than a Python loop over bins: bin
    membership, per-bin sample counts, and per-bin sums of both the indicator
    and the probability are each computed in a single O(n) pass, then combined
    into per-bin accuracy/confidence and the final sample-weighted gap - an
    O(n_bins) reduction, not O(n_bins * n).

    Args:
        y_bin (np.ndarray): Binary indicator (0/1) of shape (n_samples,).
        p (np.ndarray): Predicted probability of the positive class, same shape.
        n_bins (int): Number of equally-spaced bins in [0, 1].

    Returns:
        float: Expected Calibration Error in [0, 1].
    """
    n_samples = len(y_bin)
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    bin_ids = np.digitize(p, bin_edges[1:-1])
    bin_counts = np.bincount(bin_ids, minlength=n_bins).astype(float)
    nonempty = bin_counts > 0

    bin_conf = (
        np.bincount(bin_ids, weights=p, minlength=n_bins)[nonempty]
        / bin_counts[nonempty]
    )
    bin_acc = (
        np.bincount(bin_ids, weights=y_bin, minlength=n_bins)[nonempty]
        / bin_counts[nonempty]
    )
    bin_weights = bin_counts[nonempty] / n_samples
    return float(np.sum(bin_weights * np.abs(bin_acc - bin_conf)))


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
    if len(classes) == 2:
        y_bin = (y_true == classes[1]).astype(int)
        return _binary_ece(y_bin, probas[:, 1], n_bins)
    return float(
        np.mean(
            [
                _binary_ece((y_true == cls).astype(int), probas[:, i], n_bins)
                for i, cls in enumerate(classes)
            ]
        )
    )


def _brier(y_true: np.ndarray, probas: np.ndarray, classes: np.ndarray) -> float:
    """Compute the Brier score via the shared metrics implementation, label-agnostic.

    ``metrics.py``'s ``_brier_score`` binary shortcut delegates directly to
    scikit-learn's ``brier_score_loss(y_true, probas[:, 1])``, which raises
    unless ``y_true`` already takes values in {0, 1} or {-1, 1} (it has no way
    to infer which of two arbitrary labels, e.g. "cat"/"dog", is positive).
    Class labels reaching this module are not guaranteed to be 0/1, so the
    binary case is pre-binarized against ``classes[1]`` - the same "positive
    class" convention used throughout this module - before delegating. The
    multiclass OVR branch already binarizes internally per class and needs no
    such adjustment.

    Args:
        y_true (np.ndarray): True labels of shape (n_samples,).
        probas (np.ndarray): Predicted probabilities of shape (n_samples, n_classes).
        classes (np.ndarray): Unique class labels, sorted to match probas' columns.

    Returns:
        float: Brier score (lower is better).
    """
    if len(classes) == 2:
        return float(_brier_score((y_true == classes[1]).astype(int), probas[:, 1]))
    return float(_brier_score(y_true, probas))


def _eval_calibrator_oof(
    oof_probas: np.ndarray,
    y_oof: np.ndarray,
    method: str,
    classes: np.ndarray,
    fold_splits: list[tuple[np.ndarray, np.ndarray]],
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
        fold_splits (list[tuple[np.ndarray, np.ndarray]]): Cached (train_idx, val_idx)
            pairs - reusing the outer CV's own fold boundaries as the inner split
            (rather than re-deriving them) guarantees every OOF row is scored by a
            calibrator that never saw it, without a second, independently-shuffled CV.

    Returns:
        np.ndarray: Calibrated probabilities of shape (n_samples, n_classes) where
            each row was produced by a calibrator that never saw that row during fit.
    """
    cal_probas = np.zeros_like(oof_probas)
    for train_idx, val_idx in fold_splits:
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
            seed (int): Random seed forwarded to the winning CalibratedModel's
                hold-out split. Defaults to 17.
            id2label (dict[int, str] | None): Optional class-index to label mapping
                for plot titles. Defaults to None.
            selection_metric_name (str | None): Metric used to choose the best
                calibration method. Accepts "ECE", "Brier"/"Brier Score" (all always
                computed internally), None (falls back to the cross_validator's
                metric), or any key from METRICS (computed on OOF probabilities and
                added as an extra column). Defaults to "Brier Score".
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
        X, y = self._coerce_inputs(X, y)
        fold_splits = list(self.cross_validator.cv.split(X, y))

        y_oof, oof_probas = self._run_outer_cv_and_collect_oof(X, y)
        rows, calibrators = self._score_all_candidates(y_oof, oof_probas, fold_splits)
        self.calibration_scores_df_ = pd.DataFrame(rows).set_index("Method")

        sel_col, sel_direction = self._resolve_selection_column(y_oof)
        self._finalize_best_method(sel_col, sel_direction, calibrators, y, fold_splits)

        self._refit_full_estimator(X, y, eval_set)
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Return calibrated class probabilities using the full-data-fitted estimator.

        Args:
            X (pd.DataFrame): Features to predict (raw, unprocessed).

        Returns:
            np.ndarray: Calibrated probabilities of shape (n_samples, n_classes). When
                ``best_method_`` is "Uncalibrated", these are the base estimator's raw
                probabilities, returned unmodified.

        Raises:
            RuntimeError: If called before fit().
        """
        if self._fitted_full_estimator is None:
            raise RuntimeError("Call fit() before predict_proba().")

        x_proc = CrossValidationTrainer._transform_with_pipeline(
            self._fitted_full_pipeline, X
        )
        x_ord = CrossValidationTrainer._order_x_for_estimator(
            x_proc, self._fitted_full_estimator
        )
        raw = self._fitted_full_estimator.predict_proba(x_ord)
        if raw.ndim == 1:
            raw = np.column_stack([1 - raw, raw])

        if self.best_method_ == _UNCALIBRATED:
            return raw
        return _apply_calibrator(
            self.best_calibrator_, raw, self._classes, self.best_method_.lower()
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

        Each panel shows the reliability curve against the diagonal of perfect
        calibration, with the gap between them shaded, a per-bin sample-count
        histogram beneath it, and the method's ECE/Brier annotated directly.
        Displays inline automatically in a notebook - no `%matplotlib inline`
        needed - and via a normal window when run as a plain script.

        Binary: one panel per method in a single row.
        Multiclass: grid of (methods x classes) panels.

        Raises:
            RuntimeError: If called before fit().
        """
        if not self._oof_probas_by_method:
            raise RuntimeError("Call fit() before plot_calibration_curves().")

        y_true = self._y_oof
        method_names = list(self._oof_probas_by_method.keys())
        n_methods = len(method_names)

        if len(self._classes) == 2:
            fig, axes = plt.subplots(
                1,
                n_methods,
                figsize=(4.6 * n_methods, 5.2),
                squeeze=False,
                facecolor=_PLOT_SURFACE,
                gridspec_kw={"wspace": 0.35},
            )
            y_bin = (y_true == self._classes[1]).astype(int)
            for col, name in enumerate(method_names):
                ax = axes[0, col]
                hist_ax = self._draw_reliability_panel(
                    ax, y_bin, self._oof_probas_by_method[name][:, 1], name, name
                )
                hist_ax.set_xlabel(
                    "Mean predicted probability", fontsize=9, color=_PLOT_MUTED_INK
                )
            fig.suptitle(
                "Calibration Reliability Diagrams",
                fontsize=15,
                fontweight="bold",
                color=_PLOT_INK,
                y=0.99,
            )
            fig.subplots_adjust(top=0.82, bottom=0.24)
        else:
            n_classes = len(self._classes)
            labels = self._resolve_labels(n_classes, self.id2label)
            fig, axes = plt.subplots(
                n_methods,
                n_classes,
                figsize=(4.2 * n_classes, 4.6 * n_methods),
                squeeze=False,
                facecolor=_PLOT_SURFACE,
                gridspec_kw={"wspace": 0.3, "hspace": 0.55},
            )
            for row, name in enumerate(method_names):
                probas = self._oof_probas_by_method[name]
                for col, cls in enumerate(self._classes):
                    ax = axes[row, col]
                    y_bin = (y_true == cls).astype(int)
                    panel_title = f"Class {labels[col]}"
                    hist_ax = self._draw_reliability_panel(
                        ax, y_bin, probas[:, col], panel_title, name
                    )
                    if col == 0:
                        ax.annotate(
                            name,
                            xy=(0, 0.5),
                            xytext=(-ax.yaxis.labelpad - 30, 0),
                            xycoords=ax.yaxis.label,
                            textcoords="offset points",
                            fontsize=11,
                            fontweight="bold",
                            color=(
                                _PLOT_WIN if name == self.best_method_ else _PLOT_INK
                            ),
                            ha="right",
                            va="center",
                            rotation=90,
                        )
                    if row == n_methods - 1:
                        hist_ax.set_xlabel(
                            "Mean predicted probability",
                            fontsize=9,
                            color=_PLOT_MUTED_INK,
                        )
            fig.suptitle(
                "Calibration Reliability Diagrams (Multiclass)",
                fontsize=15,
                fontweight="bold",
                color=_PLOT_INK,
                y=0.995,
            )
            fig.subplots_adjust(top=0.90, bottom=0.14, left=0.13)

        _show_figure(fig)

    def style_scores(self) -> Styler:
        """Return calibration_scores_df_ as a fancy-styled table for notebook display.

        Applies this package's standard report styling (see
        ``kvbiii_ml.evaluation.generate_reports``), with the row for
        ``best_method_`` bolded and marked with a green accent border.

        Returns:
            Styler: Styled table - display it directly as a notebook cell's
                last expression, or via ``IPython.display.display(...)``.

        Raises:
            RuntimeError: If called before fit().
        """
        if self.calibration_scores_df_.empty:
            raise RuntimeError("Call fit() before style_scores().")

        styled = _apply_fancy_styling(
            self.calibration_scores_df_.style.format("{:.4f}"),
            "🎯 Calibration Method Comparison",
        )
        return styled.apply(self._highlight_best_row, axis=1)

    def _draw_reliability_panel(
        self,
        ax: Axes,
        y_bin: np.ndarray,
        p: np.ndarray,
        title: str,
        method_name: str,
    ) -> Axes:
        """Draw one reliability curve with a miscalibration-gap fill and bin-density inset.

        Args:
            ax (Axes): Axes to draw the reliability curve into.
            y_bin (np.ndarray): Binary indicator (0/1) for the positive class shown.
            p (np.ndarray): Predicted probability of that positive class.
            title (str): Panel title (method name, or class label in a multiclass grid).
            method_name (str): Calibration method this panel belongs to, used only to
                decide whether to apply the "best method" highlight treatment.

        Returns:
            Axes: The bin-density inset axes, so callers can attach the shared
                x-axis label beneath it instead of beneath the curve above it.
        """
        frac_pos, mean_pred = calibration_curve(
            y_bin, p, n_bins=self.n_bins, strategy="uniform"
        )

        ax.plot(
            [0, 1],
            [0, 1],
            color=_PLOT_REFERENCE,
            linestyle="--",
            linewidth=1.4,
            zorder=1,
        )
        ax.fill_between(
            mean_pred, frac_pos, mean_pred, color=_PLOT_ACCENT, alpha=0.16, zorder=2
        )
        ax.plot(
            mean_pred,
            frac_pos,
            color=_PLOT_ACCENT,
            linewidth=2.2,
            marker="o",
            markersize=6.5,
            markerfacecolor="white",
            markeredgewidth=1.6,
            zorder=3,
        )

        ece = _binary_ece(y_bin, p, self.n_bins)
        brier = float(_brier_score(y_bin, p))
        ax.text(
            0.04,
            0.96,
            f"ECE {ece:.3f}\nBrier {brier:.3f}",
            transform=ax.transAxes,
            va="top",
            ha="left",
            fontsize=8.5,
            color=_PLOT_MUTED_INK,
            bbox={
                "boxstyle": "round,pad=0.3",
                "facecolor": "white",
                "edgecolor": _PLOT_GRID,
                "alpha": 0.9,
            },
        )

        hist_ax = ax.inset_axes([0, -0.24, 1, 0.16], sharex=ax)
        bin_edges = np.linspace(0.0, 1.0, self.n_bins + 1)
        counts, _ = np.histogram(p, bins=bin_edges)
        hist_ax.bar(
            (bin_edges[:-1] + bin_edges[1:]) / 2,
            counts,
            width=(bin_edges[1] - bin_edges[0]) * 0.9,
            color=_PLOT_ACCENT_TINT,
            edgecolor="white",
            linewidth=0.5,
        )
        hist_ax.set_yticks([])
        hist_ax.set_xlim(0, 1)
        for spine in ("top", "right", "left"):
            hist_ax.spines[spine].set_visible(False)
        hist_ax.tick_params(axis="x", labelsize=7.5, colors=_PLOT_MUTED_INK)
        hist_ax.set_xlabel("")

        ax.set_xlim(-0.02, 1.02)
        ax.set_ylim(-0.02, 1.02)
        ax.set_ylabel("Fraction of positives", fontsize=9, color=_PLOT_MUTED_INK)
        ax.grid(color=_PLOT_GRID, linewidth=0.8, zorder=0)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)
        for spine in ("bottom", "left"):
            ax.spines[spine].set_color(_PLOT_GRID)
        ax.tick_params(colors=_PLOT_MUTED_INK, labelsize=8.5)
        ax.set_xticklabels([])

        is_best = method_name == self.best_method_
        ax.set_title(
            f"★ {title}" if is_best else title,
            fontsize=11,
            fontweight="bold",
            color=_PLOT_WIN if is_best else _PLOT_INK,
            pad=10,
        )
        return hist_ax

    @staticmethod
    def _coerce_inputs(X: pd.DataFrame, y: pd.Series) -> tuple[pd.DataFrame, pd.Series]:
        """Coerce X/y to pandas containers when raw array-likes are passed.

        Args:
            X (pd.DataFrame): Feature matrix, or an array-like coercible to one.
            y (pd.Series): Target vector, or an array-like coercible to one.

        Returns:
            tuple[pd.DataFrame, pd.Series]: X and y as pandas containers.
        """
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)
        if not isinstance(y, pd.Series):
            y = pd.Series(np.asarray(y).ravel())
        return X, y

    def _run_outer_cv_and_collect_oof(
        self, X: pd.DataFrame, y: pd.Series
    ) -> tuple[np.ndarray, np.ndarray]:
        """Run the outer cross-validation once and collect out-of-fold probabilities.

        Reuses the fold estimators/pipelines cross_validator.fit() already produced
        to derive OOF probabilities, rather than re-fitting.

        Args:
            X (pd.DataFrame): Full training feature matrix.
            y (pd.Series): Full training target vector.

        Returns:
            tuple[np.ndarray, np.ndarray]: OOF true labels of shape (n_samples,) and
                OOF probabilities of shape (n_samples, n_classes).
        """
        train_scores, valid_scores, _ = self.cross_validator.fit(self.estimator, X, y)
        self.train_scores_ = list(train_scores)
        self.valid_scores_ = list(valid_scores)

        y_oof, oof_probas = _collect_oof_probas(self.cross_validator, X, y)
        self._y_oof = y_oof
        self._classes = np.unique(y_oof)
        return y_oof, oof_probas

    def _score_all_candidates(
        self,
        y_oof: np.ndarray,
        oof_probas: np.ndarray,
        fold_splits: list[tuple[np.ndarray, np.ndarray]],
    ) -> tuple[list[dict], dict[str, IsotonicRegression | LogisticRegression | list]]:
        """Score the uncalibrated baseline and every configured calibration method.

        Args:
            y_oof (np.ndarray): OOF true labels of shape (n_samples,).
            oof_probas (np.ndarray): OOF probabilities from the base estimator.
            fold_splits (list[tuple[np.ndarray, np.ndarray]]): Cached outer-CV fold
                (train_idx, val_idx) pairs, reused as the inner CV for honest
                per-method calibrator evaluation.

        Returns:
            tuple[list[dict], dict[str, IsotonicRegression | LogisticRegression | list]]:
                One score row per candidate (Uncalibrated plus every method), and a
                method-name-to-calibrator mapping (each fitted on the full OOF set).
        """
        metric_fn = self.cross_validator.metric_fn
        metric_type = self.cross_validator.metric_type

        def _score_row(name: str, probas: np.ndarray) -> dict:
            preds = self._probas_to_metric_input(probas, metric_type, self._classes)
            return {
                "Method": name,
                self.cross_validator.metric_name: float(metric_fn(y_oof, preds)),
                "ECE": _compute_ece(y_oof, probas, self.n_bins, self._classes),
                "Brier": _brier(y_oof, probas, self._classes),
            }

        rows = [_score_row(_UNCALIBRATED, oof_probas)]
        self._oof_probas_by_method[_UNCALIBRATED] = oof_probas

        calibrators: dict[str, IsotonicRegression | LogisticRegression | list] = {}
        for method in self.methods:
            eval_probas = _eval_calibrator_oof(
                oof_probas, y_oof, method, self._classes, fold_splits
            )
            calibrators[method] = _fit_calibrator(
                oof_probas, y_oof, method, self._classes
            )
            method_name = method.capitalize()
            rows.append(_score_row(method_name, eval_probas))
            self._oof_probas_by_method[method_name] = eval_probas

        return rows, calibrators

    def _resolve_selection_column(self, y_oof: np.ndarray) -> tuple[str, str]:
        """Determine which column of calibration_scores_df_ drives method selection.

        Computes and appends an extra column only when selection_metric_name is a
        METRICS key not already present as the cross-validator's own metric or one
        of the always-computed ECE/Brier columns, so it is never scored twice.

        Args:
            y_oof (np.ndarray): OOF true labels, used to score an extra column.

        Returns:
            tuple[str, str]: The calibration_scores_df_ column to select on, and its
                optimization direction, either "minimize" or "maximize".
        """
        sel_name = self.selection_metric_name
        if sel_name is None or sel_name == self.cross_validator.metric_name:
            return (
                self.cross_validator.metric_name,
                self.cross_validator.metric_direction,
            )
        if sel_name in _BUILTIN_SELECTION_COLS:
            return _BUILTIN_SELECTION_COLS[sel_name]

        sel_fn = get_metric_function(sel_name)
        sel_type = get_metric_type(sel_name)
        self.calibration_scores_df_[sel_name] = pd.Series(
            {
                method_key: float(
                    sel_fn(
                        y_oof,
                        self._probas_to_metric_input(probas, sel_type, self._classes),
                    )
                )
                for method_key, probas in self._oof_probas_by_method.items()
            }
        )
        return sel_name, get_metric_direction(sel_name)

    def _finalize_best_method(
        self,
        sel_col: str,
        sel_direction: str,
        calibrators: dict[str, IsotonicRegression | LogisticRegression | list],
        y: pd.Series,
        fold_splits: list[tuple[np.ndarray, np.ndarray]],
    ) -> None:
        """Pick the winning method and populate every best_* attribute.

        Args:
            sel_col (str): calibration_scores_df_ column to select the winner by.
            sel_direction (str): "minimize" or "maximize".
            calibrators (dict[str, IsotonicRegression | LogisticRegression | list]):
                Method name to its calibrator fitted on the full OOF set.
            y (pd.Series): Full training target, used to score calibrated_valid_scores_.
            fold_splits (list[tuple[np.ndarray, np.ndarray]]): Cached outer-CV folds.
        """
        scores = self.calibration_scores_df_[sel_col]
        best_row = str(
            scores.idxmax() if sel_direction == "maximize" else scores.idxmin()
        )
        self.best_method_ = best_row

        best_probas = self._oof_probas_by_method[best_row]
        metric_fn = self.cross_validator.metric_fn
        metric_type = self.cross_validator.metric_type
        y_arr = np.asarray(y)
        self.calibrated_valid_scores_ = [
            float(
                metric_fn(
                    y_arr[val_idx],
                    self._probas_to_metric_input(
                        best_probas[val_idx], metric_type, self._classes
                    ),
                )
            )
            for _, val_idx in fold_splits
        ]

        if best_row == _UNCALIBRATED:
            self.best_calibrator_ = None
            self.best_estimator_ = self.estimator
            return

        best_cal_method = best_row.lower()
        self.best_calibrator_ = calibrators[best_cal_method]
        self.best_estimator_ = CalibratedModel(
            estimator=deepcopy(self.estimator),
            problem_type="classification",
            calibration_method=best_cal_method,
            calibration_size=self.calibration_holdout_size,
            seed=self.seed,
        )

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
        pipeline = self.cross_validator.preprocessing_pipeline
        fitted_pipeline = clone(pipeline) if pipeline is not None else None
        x_proc = (
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
            final_est, x_proc, y, X_valid, y_valid
        )

    def _highlight_best_row(self, row: pd.Series) -> list[str]:
        """Bold the row matching best_method_ with a green accent border.

        Args:
            row (pd.Series): One row of calibration_scores_df_.

        Returns:
            list[str]: Per-cell CSS strings for this row.
        """
        if row.name != self.best_method_:
            return [""] * len(row)
        return [f"font-weight: 800; border-left: 4px solid {_PLOT_WIN};"] * len(row)

    @staticmethod
    def _probas_to_metric_input(
        probas: np.ndarray, metric_type: str, classes: np.ndarray
    ) -> np.ndarray:
        """Convert probability matrix to the format expected by the metric function.

        Args:
            probas (np.ndarray): Probabilities of shape (n_samples, n_classes).
            metric_type (str): Either "probs" or "preds".
            classes (np.ndarray): Unique class labels, ordered to match probas'
                columns - required to map an argmax column index back to the
                actual label value for "preds" metrics (e.g. "Accuracy", "F1"),
                since class labels are not guaranteed to be 0..n_classes-1.

        Returns:
            np.ndarray: Positive-class column for binary "probs", full matrix for
                multiclass "probs", or predicted class labels for "preds".
        """
        if metric_type == "probs":
            return probas[:, 1] if probas.shape[1] == 2 else probas
        return classes[np.argmax(probas, axis=1)]

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
    import time

    import numpy as np
    import pandas as pd
    from lightgbm import LGBMClassifier
    from catboost import CatBoostClassifier
    from sklearn.datasets import make_classification
    from sklearn.model_selection import StratifiedKFold, train_test_split
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
        x_arr, y_arr = make_classification(
            n_samples=N_SAMPLES,
            n_features=N_FEATURES,
            n_informative=6,
            n_redundant=2,
            n_classes=n_classes,
            n_clusters_per_class=1,
            random_state=RANDOM_STATE,
        )
        return pd.DataFrame(x_arr, columns=FEATURE_NAMES), pd.Series(
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

    def _run_demo() -> None:
        """Run the ClassificationCalibrator demo scenarios."""
        print("=" * 75)
        print("Scenario 1: Binary + LightGBM + preprocessing pipeline")
        print("=" * 75)
        x_bin, y_bin = _make_clf_data(n_classes=2)
        lgbm_bin = LGBMClassifier(
            n_estimators=200,
            early_stopping_rounds=ES,
            verbose=-1,
            random_state=RANDOM_STATE,
        )
        cv_1 = _build_cv("Log Loss", _build_pipeline())
        cal_1 = ClassificationCalibrator(estimator=lgbm_bin, cross_validator=cv_1)
        cal_1.fit(x_bin, y_bin)
        print(f"Best method: {cal_1.best_method_}")
        print(cal_1.calibration_scores_df_)
        print("Passing best_estimator_ to cross_validator.fit():")
        _, valid_s, _ = cv_1.fit(cal_1.best_estimator_, x_bin, y_bin)
        print(f"  Calibrated CV valid: {np.mean(valid_s):.4f} ± {np.std(valid_s):.4f}")
        cal_1.plot_calibration_curves()

        print("\n" + "=" * 75)
        print("Scenario 2: Binary + CatBoost with categorical dtype columns")
        print("=" * 75)
        try:
            rng = np.random.RandomState(RANDOM_STATE)
            x_cat = x_bin.copy()
            x_cat["cat_feat"] = pd.Categorical(
                rng.choice(["A", "B", "C"], size=N_SAMPLES)
            )
            cat_cb = CatBoostClassifier(
                iterations=200,
                early_stopping_rounds=ES,
                verbose=0,
                random_state=RANDOM_STATE,
            )
            cv_2 = _build_cv("Log Loss")
            for label, est, x_in in [
                ("LightGBM", lgbm_bin, x_bin),
                ("CatBoost", cat_cb, x_cat),
            ]:
                cal_2 = ClassificationCalibrator(estimator=est, cross_validator=cv_2)
                cal_2.fit(x_in, y_bin)
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
        cal_3.fit(x_bin, y_bin)
        print(f"  best_method_: {cal_3.best_method_}")
        print(
            f"  best_estimator_ is CalibratedModel: {isinstance(cal_3.best_estimator_, CalibratedModel)}"
        )
        print(cal_3.calibration_scores_df_)

        start = time.perf_counter()
        _, valid_s_3, _ = cv_3.fit(cal_3.best_estimator_, x_bin, y_bin)
        print(
            f"  Time for CV fit with best_estimator_: {time.perf_counter() - start:.2f} seconds"
        )
        print(
            f"  Calibrated CV valid: {np.mean(valid_s_3):.4f} ± {np.std(valid_s_3):.4f}"
        )
        cal_3.plot_calibration_curves()

        print("\n" + "=" * 75)
        print(
            "Scenario 4: Multiclass + LightGBM + eval_set for early stopping on final refit"
        )
        print("=" * 75)
        x_mc, y_mc = _make_clf_data(n_classes=3)
        lgbm_mc = LGBMClassifier(
            n_estimators=200,
            early_stopping_rounds=ES,
            verbose=-1,
            random_state=RANDOM_STATE,
        )
        cv_4 = _build_cv("Brier Score", _build_pipeline())
        cal_4 = ClassificationCalibrator(estimator=lgbm_mc, cross_validator=cv_4)

        x_tr, x_val, y_tr, y_val = train_test_split(
            x_mc, y_mc, test_size=0.2, random_state=RANDOM_STATE, stratify=y_mc
        )
        cal_4.fit(x_tr, y_tr, eval_set=[(x_val, y_val)])
        print(f"  best_method_: {cal_4.best_method_}")
        print(cal_4.calibration_scores_df_)
        print("Passing best_estimator_ to cross_validator.fit():")
        _, valid_s_mc, _ = cv_4.fit(cal_4.best_estimator_, x_mc, y_mc)
        print(
            f"  Calibrated CV valid: {np.mean(valid_s_mc):.4f} ± {np.std(valid_s_mc):.4f}"
        )
        cal_4.plot_calibration_curves()

    _run_demo()
