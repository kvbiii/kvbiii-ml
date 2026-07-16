from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde
from sklearn.base import BaseEstimator

sys.path.append(str(Path(__file__).resolve().parents[3]))
from kvbiii_ml.modeling.training.cross_validation import CrossValidationTrainer
from kvbiii_ml.modeling.training.oof_model import _collect_oof_probas

_VALID_THRESHOLD_METHODS = frozenset({"auto", "top_k", "fixed"})
_DEFAULT_PERCENTILE = 80.0
_DEFAULT_TOP_K_PCT = 0.20
_DEFAULT_FIXED_THRESHOLD = 0.90
_CANDIDATE_COLOR = "#5dade2"
_SELECTED_COLOR = "#2ecc71"
_THRESHOLD_COLOR = "#2c3e50"
_ORIGINAL_COLOR = "#3498db"
_PSEUDO_COLOR = "#9b59b6"
_KDE_GRID_POINTS = 400
_MIN_KDE_SAMPLES = 5
_SUPTITLE_FONTSIZE = 14
_TITLE_FONTSIZE = 12
_LABEL_FONTSIZE = 11
_TICK_FONTSIZE = 10
_LEGEND_FONTSIZE = 10
_ANNOTATION_FONTSIZE = 10
_THRESHOLD_HARD_CAP = 1.0 - 1e-6


class PseudoLabelGenerator:
    """Generate pseudo labels for unlabeled data using a cross-validated model.

    Trains a model on labeled data via cross-validation, collects out-of-fold
    probabilities to calibrate per-class confidence thresholds, then applies those
    thresholds to select high-confidence predictions on unlabeled data as pseudo
    labels.  The result is an augmented dataset ready for retraining.

    Three threshold strategies are supported:

    - ``"auto"``: per-class threshold = ``min(T_oof, T_unl)`` where T_oof is the
      ``threshold_percentile``-th percentile of OOF class-specific probabilities for
      true class-c samples, and T_unl is the same percentile of unlabeled predictions
      for class c.  Taking the minimum prevents the float64 saturation trap (where
      gradient boosters output exactly 1.0 on small OOF folds but slightly below on
      large unlabeled batches) and gracefully handles train/unlabeled distribution
      shift where OOF confidences are systematically higher.
    - ``"top_k"``: per-class, take the ``top_k_pct`` fraction of unlabeled samples
      with the highest predicted probability for that class (among those whose argmax
      prediction is that class).
    - ``"fixed"``: single scalar threshold applied to all classes.

    Pseudo-labeled samples are weighted by their predicted confidence; original
    labeled samples receive weight 1.0.
    """

    def __init__(
        self,
        cross_validator: CrossValidationTrainer,
        threshold_method: str = "auto",
        threshold_percentile: float = _DEFAULT_PERCENTILE,
        top_k_pct: float = _DEFAULT_TOP_K_PCT,
        fixed_threshold: float = _DEFAULT_FIXED_THRESHOLD,
    ) -> None:
        """Initialize the pseudo-label generator.

        Args:
            cross_validator (CrossValidationTrainer): Configured CV trainer. Its
                fitted pipelines are reused when predicting on unlabeled data.
            threshold_method (str): One of "auto", "top_k", or "fixed".
                Defaults to "auto".
            threshold_percentile (float): Used by "auto". Percentile (0–100) of the
                OOF class-specific probability distribution for each class used to set
                the threshold. Higher = stricter = fewer but more reliable pseudo labels.
                Defaults to 80.
            top_k_pct (float): Used by "top_k". Fraction of unlabeled samples to
                keep per class (e.g. 0.2 = top-20%). Defaults to 0.20.
            fixed_threshold (float): Used by "fixed". Confidence cutoff applied to
                all classes. Defaults to 0.90.

        Raises:
            ValueError: If threshold_method is not one of the supported values.
            ValueError: If threshold_percentile is outside [0, 100].
            ValueError: If top_k_pct is outside (0, 1].
            ValueError: If fixed_threshold is outside (0, 1).
        """
        if threshold_method not in _VALID_THRESHOLD_METHODS:
            raise ValueError(
                f"threshold_method must be one of {sorted(_VALID_THRESHOLD_METHODS)}, "
                f"got '{threshold_method}'."
            )
        if not 0.0 <= threshold_percentile <= 100.0:
            raise ValueError("threshold_percentile must be in [0, 100].")
        if not 0.0 < top_k_pct <= 1.0:
            raise ValueError("top_k_pct must be in (0, 1].")
        if not 0.0 < fixed_threshold < 1.0:
            raise ValueError("fixed_threshold must be in (0, 1).")

        self.cross_validator = cross_validator
        self.threshold_method = threshold_method
        self.threshold_percentile = threshold_percentile
        self.top_k_pct = top_k_pct
        self.fixed_threshold = fixed_threshold

        self.classes_: np.ndarray | None = None
        self.thresholds_: dict[int | str, float] = {}
        self.pseudo_labels_df_: pd.DataFrame = pd.DataFrame()
        self._x_train: pd.DataFrame | None = None
        self._y_train: pd.Series | None = None
        self._unlabeled_probas: np.ndarray | None = None

    def fit(
        self,
        estimator: BaseEstimator,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        x_unlabeled: pd.DataFrame,
    ) -> "PseudoLabelGenerator":
        """Fit the model via CV, compute thresholds, and generate pseudo labels.

        Args:
            estimator (BaseEstimator): Unfitted estimator to train.
            X_train (pd.DataFrame): Labeled training features.
            y_train (pd.Series): Labeled training target.
            x_unlabeled (pd.DataFrame): Unlabeled features to pseudo-label.

        Returns:
            PseudoLabelGenerator: Fitted self.
        """
        if not isinstance(X_train, pd.DataFrame):
            X_train = pd.DataFrame(X_train)
        if not isinstance(y_train, pd.Series):
            y_train = pd.Series(np.asarray(y_train))
        if not isinstance(x_unlabeled, pd.DataFrame):
            x_unlabeled = pd.DataFrame(x_unlabeled)

        self._x_train = X_train
        self._y_train = y_train
        self.classes_ = np.unique(y_train)

        self.cross_validator.fit(estimator, X_train, y_train)

        self._unlabeled_probas = self.cross_validator.predict_proba(x_unlabeled)
        if self._unlabeled_probas.ndim == 1:
            self._unlabeled_probas = np.column_stack(
                [1 - self._unlabeled_probas, self._unlabeled_probas]
            )

        argmax_indices = np.argmax(self._unlabeled_probas, axis=1)
        max_confidences = self._unlabeled_probas.max(axis=1)

        if self.threshold_method == "auto":
            y_oof, oof_probas = _collect_oof_probas(
                self.cross_validator, X_train, y_train
            )
            self.thresholds_ = self._compute_thresholds_auto(oof_probas, y_oof)
        elif self.threshold_method == "top_k":
            self.thresholds_ = self._compute_thresholds_top_k(
                argmax_indices, max_confidences
            )
        else:
            self.thresholds_ = {cls: self.fixed_threshold for cls in self.classes_}

        selected_mask, pseudo_labels, confidences = self._select_pseudo_labels(
            argmax_indices, max_confidences
        )

        self.pseudo_labels_df_ = x_unlabeled.copy()
        self.pseudo_labels_df_["_pseudo_label"] = np.where(
            selected_mask, pseudo_labels, np.nan
        )
        self.pseudo_labels_df_["_confidence"] = np.where(
            selected_mask, confidences, np.nan
        )
        self.pseudo_labels_df_["_selected"] = selected_mask
        self.pseudo_labels_df_ = self.pseudo_labels_df_[selected_mask].copy()
        self.pseudo_labels_df_["_pseudo_label"] = self.pseudo_labels_df_[
            "_pseudo_label"
        ].astype(y_train.dtype)

        self._print_summary(selected_mask, pseudo_labels, confidences)
        return self

    def get_augmented_dataset(self) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
        """Return the combined labeled + pseudo-labeled dataset.

        Original labeled samples receive sample weight 1.0. Pseudo-labeled
        samples receive weight equal to their predicted confidence.

        Returns:
            tuple[pd.DataFrame, pd.Series, pd.Series]: Feature matrix, target
                vector, and sample weights - all aligned row-wise.

        Raises:
            RuntimeError: If called before fit().
        """
        if self._x_train is None:
            raise RuntimeError("Call fit() before get_augmented_dataset().")

        feature_cols = [
            c for c in self.pseudo_labels_df_.columns if not c.startswith("_")
        ]
        x_pseudo = self.pseudo_labels_df_[feature_cols].reset_index(drop=True)
        y_pseudo = self.pseudo_labels_df_["_pseudo_label"].reset_index(drop=True)
        w_pseudo = self.pseudo_labels_df_["_confidence"].reset_index(drop=True)

        x_combined = pd.concat(
            [self._x_train.reset_index(drop=True), x_pseudo], ignore_index=True
        )
        y_combined = pd.concat(
            [self._y_train.reset_index(drop=True), y_pseudo], ignore_index=True
        )
        w_combined = pd.concat(
            [
                pd.Series(np.ones(len(self._x_train)), name="_weight", dtype=float),
                w_pseudo.rename("_weight"),
            ],
            ignore_index=True,
        )
        return x_combined, y_combined, w_combined

    def plot_pseudo_label_stats(self) -> None:
        """Visualise pseudo-label confidence distributions and class composition.

        Produces two figures:

        - Confidence-score density per predicted class, with the selection
          threshold marked and the selected region shaded, annotated with how
          many of that class's candidates were kept.
        - Class composition shift: normalised proportions of original vs
          pseudo labels, plus the absolute pseudo-label volume per class.

        Raises:
            RuntimeError: If called before fit().
        """
        if self._unlabeled_probas is None:
            raise RuntimeError("Call fit() before plot_pseudo_label_stats().")

        self._plot_confidence_distributions()
        self._plot_label_distribution()

    def _compute_thresholds_auto(
        self, oof_probas: np.ndarray, y_oof: np.ndarray
    ) -> dict[int | str, float]:
        """Compute per-class thresholds using OOF quality gate and unlabeled anchor.

        For each class ``c`` at column index ``i``:

        1. **OOF quality gate** (T_oof): ``threshold_percentile``-th percentile of the
           model's class-``c`` probability on all ground-truth class-``c`` OOF samples
           (true positives, including misclassified ones).  Including uncertain true
           positives pulls the distribution below 1.0 and reflects a more realistic
           confidence requirement.

        2. **Unlabeled anchor** (T_unl): ``threshold_percentile``-th percentile of the
           model's class-``c`` probability on unlabeled samples predicted as class ``c``.
           This adapts the threshold to the actual distribution of unlabeled predictions.

        3. **Final threshold**: ``min(T_oof, T_unl)`` — ensures the threshold never
           exceeds what is naturally achievable on the unlabeled data.  Handles two
           common failure modes: (a) float64 saturation, where gradient boosters output
           exactly 1.0 on small OOF sub-folds but slightly below on large unlabeled
           batches; (b) train/unlabeled distribution shift, where the model is
           systematically more confident on OOF than on unlabeled data.

        The hard cap at ``_THRESHOLD_HARD_CAP = 1 - 1e-6`` still applies as a backstop.

        Args:
            oof_probas (np.ndarray): OOF probabilities of shape (n_samples, n_classes).
            y_oof (np.ndarray): OOF true labels.

        Returns:
            dict[int | str, float]: Per-class confidence thresholds in (0, 1).
        """
        argmax_oof = np.argmax(oof_probas, axis=1)
        argmax_unl = np.argmax(self._unlabeled_probas, axis=1)
        thresholds = {}

        for i, cls in enumerate(self.classes_):
            tp_mask = y_oof == cls
            if tp_mask.sum() >= 10:
                oof_class_probs = oof_probas[tp_mask, i]
            else:
                pred_mask = argmax_oof == i
                oof_class_probs = (
                    oof_probas[pred_mask, i] if pred_mask.any() else np.array([0.5])
                )

            t_oof = min(
                float(np.percentile(oof_class_probs, self.threshold_percentile)),
                _THRESHOLD_HARD_CAP,
            )

            unl_cls_mask = argmax_unl == i
            t_unl = (
                float(
                    np.percentile(
                        self._unlabeled_probas[unl_cls_mask, i],
                        self.threshold_percentile,
                    )
                )
                if unl_cls_mask.any()
                else t_oof
            )

            thresholds[cls] = min(t_oof, t_unl)

        return thresholds

    def _compute_thresholds_top_k(
        self,
        argmax_indices: np.ndarray,
        max_confidences: np.ndarray,
    ) -> dict[int | str, float]:
        """Compute per-class thresholds that yield the top-k% of unlabeled samples.

        For each class, considers only samples whose argmax prediction is that class,
        then sets the threshold at the (1 - top_k_pct)-th quantile of their
        confidence scores so that the top fraction is selected.

        Args:
            argmax_indices (np.ndarray): Argmax class indices for each unlabeled sample.
            max_confidences (np.ndarray): Max probability per unlabeled sample.

        Returns:
            dict[int | str, float]: Per-class confidence thresholds.
        """
        thresholds = {}
        for i, cls in enumerate(self.classes_):
            cls_mask = argmax_indices == i
            if not cls_mask.any():
                thresholds[cls] = 1.0
                continue
            cutoff = (1.0 - self.top_k_pct) * 100.0
            thresholds[cls] = float(np.percentile(max_confidences[cls_mask], cutoff))
        return thresholds

    def _select_pseudo_labels(
        self,
        argmax_indices: np.ndarray,
        max_confidences: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Apply per-class thresholds to select pseudo-labeled samples.

        Args:
            argmax_indices (np.ndarray): Argmax class indices per unlabeled sample.
            max_confidences (np.ndarray): Max probability per unlabeled sample.

        Returns:
            tuple[np.ndarray, np.ndarray, np.ndarray]: Boolean selection mask,
                predicted class labels for each sample, and max confidence scores.
        """
        pseudo_labels = self.classes_[argmax_indices]
        threshold_per_sample = np.array(
            [self.thresholds_[cls] for cls in pseudo_labels], dtype=float
        )
        return max_confidences >= threshold_per_sample, pseudo_labels, max_confidences

    @staticmethod
    def _density_curve(values: np.ndarray, grid: np.ndarray) -> np.ndarray:
        """Estimate a smooth density of ``values`` evaluated at ``grid``.

        Falls back to an interpolated histogram when the sample is too small
        or constant for kernel density estimation to be numerically stable.

        Args:
            values (np.ndarray): One-dimensional sample of confidence scores.
            grid (np.ndarray): Points at which to evaluate the density.

        Returns:
            np.ndarray: Density values aligned with ``grid``.
        """
        if len(values) < _MIN_KDE_SAMPLES or np.isclose(np.std(values), 0.0):
            counts, edges = np.histogram(
                values, bins=20, range=(grid[0], grid[-1]), density=True
            )
            bin_centers = (edges[:-1] + edges[1:]) / 2
            return np.interp(grid, bin_centers, counts, left=0.0, right=0.0)
        return gaussian_kde(values)(grid)

    def _plot_confidence_distributions(self) -> None:
        """Plot per-class confidence-score densities with selection cutoffs.

        For each class, draws a smoothed density of the maximum predicted
        probability among unlabeled samples whose argmax prediction is that
        class, marks the per-class threshold with a dashed line, and shades
        the region above it in green to highlight the selected slice.
        """
        n_classes = len(self.classes_)
        argmax_labels = self.classes_[np.argmax(self._unlabeled_probas, axis=1)]
        max_confidences = self._unlabeled_probas.max(axis=1)

        fig, axes = plt.subplots(
            1, n_classes, figsize=(5.5 * n_classes, 4.8), squeeze=False
        )
        for col, cls in enumerate(self.classes_):
            ax = axes[0, col]
            cls_mask = argmax_labels == cls
            confidences = max_confidences[cls_mask]
            n_candidates = int(cls_mask.sum())

            if n_candidates == 0:
                ax.set_title(f"Class {cls} - no candidates", fontsize=_TITLE_FONTSIZE)
                ax.set_axis_off()
                continue

            threshold = self.thresholds_.get(cls, 0.5)
            n_selected = int((confidences >= threshold).sum())
            selection_rate = 100 * n_selected / n_candidates

            low = max(0.0, float(np.percentile(confidences, 1)) - 0.05)
            grid = np.linspace(low, 1.0, _KDE_GRID_POINTS)
            density = self._density_curve(confidences, grid)

            ax.plot(grid, density, color=_THRESHOLD_COLOR, linewidth=1.2)
            ax.fill_between(
                grid,
                density,
                color=_CANDIDATE_COLOR,
                alpha=0.35,
                label="Candidate density",
            )
            ax.fill_between(
                grid,
                density,
                where=grid >= threshold,
                color=_SELECTED_COLOR,
                alpha=0.7,
                label=f"Selected region ({n_selected:,}/{n_candidates:,})",
            )
            ax.axvline(
                threshold,
                color=_THRESHOLD_COLOR,
                linestyle="--",
                linewidth=1.5,
                label=f"Threshold = {threshold:.3f}",
            )
            ax.set_title(
                f"Class {cls}\n{n_selected:,} of {n_candidates:,} candidates "
                f"selected ({selection_rate:.1f}%)",
                fontsize=_TITLE_FONTSIZE,
            )
            ax.set_xlabel("Max predicted probability", fontsize=_LABEL_FONTSIZE)
            ax.set_ylabel("Density" if col == 0 else "", fontsize=_LABEL_FONTSIZE)
            ax.tick_params(axis="both", labelsize=_TICK_FONTSIZE)
            ax.set_xlim(low, 1.005)
            ax.set_ylim(bottom=0)
            ax.spines[["top", "right"]].set_visible(False)
            ax.grid(axis="y", alpha=0.25)
            ax.legend(fontsize=_LEGEND_FONTSIZE, loc="upper left")

        fig.suptitle(
            f"Confidence distributions by predicted class  |  method={self.threshold_method}",
            fontsize=_SUPTITLE_FONTSIZE,
            fontweight="bold",
        )
        plt.tight_layout()
        plt.show()

    def _plot_label_distribution(self) -> None:
        """Plot original class proportions against the newly generated pseudo labels.

        A grouped bar chart contrasts the normalised class composition of the
        original labeled set with that of the pseudo-labeled set, annotated
        with each bar's share of its respective set.
        """
        n_classes = len(self.classes_)
        x = np.arange(n_classes)
        width = 0.38

        train_counts = self._y_train.value_counts(normalize=True).reindex(
            self.classes_, fill_value=0
        )
        pseudo_counts = (
            self.pseudo_labels_df_["_pseudo_label"]
            .value_counts(normalize=True)
            .reindex(self.classes_, fill_value=0)
            if not self.pseudo_labels_df_.empty
            else pd.Series(0.0, index=self.classes_)
        )

        fig, ax = plt.subplots(figsize=(8, 5.5))
        bars_original = ax.bar(
            x - width / 2,
            train_counts.values,
            width,
            label="Original labels",
            color=_ORIGINAL_COLOR,
            alpha=0.92,
            edgecolor="white",
            linewidth=1.2,
        )
        bars_pseudo = ax.bar(
            x + width / 2,
            pseudo_counts.values,
            width,
            label="Pseudo labels",
            color=_PSEUDO_COLOR,
            alpha=0.92,
            edgecolor="white",
            linewidth=1.2,
        )
        for bars, counts in (
            (bars_original, train_counts.values),
            (bars_pseudo, pseudo_counts.values),
        ):
            ax.bar_label(
                bars,
                labels=[f"{v * 100:.1f}%" for v in counts],
                fontsize=_ANNOTATION_FONTSIZE - 1,
                padding=6,
            )
        ax.set_xticks(x)
        ax.set_xticklabels([str(c) for c in self.classes_], fontsize=_TICK_FONTSIZE)
        ax.tick_params(axis="y", labelsize=_TICK_FONTSIZE)
        ax.set_ylabel("Proportion of samples", fontsize=_LABEL_FONTSIZE)
        ax.set_title(
            "Original labels vs newly generated pseudo labels", fontsize=_TITLE_FONTSIZE
        )
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", alpha=0.25)
        ax.legend(fontsize=_LEGEND_FONTSIZE, loc="lower right")
        ax.margins(y=0.15)
        fig.suptitle(
            f"Class distribution after pseudo-labeling  |  "
            f"{len(self.pseudo_labels_df_):,} pseudo labels added",
            fontsize=_SUPTITLE_FONTSIZE,
            fontweight="bold",
        )
        plt.tight_layout()
        plt.show()

    def _print_summary(
        self,
        selected_mask: np.ndarray,
        pseudo_labels: np.ndarray,
        confidences: np.ndarray,
    ) -> None:
        """Print a per-class summary of the pseudo-labeling outcome.

        Args:
            selected_mask (np.ndarray): Boolean mask of selected samples.
            pseudo_labels (np.ndarray): Predicted class for each unlabeled sample.
            confidences (np.ndarray): Max confidence score per sample.
        """
        n_total = len(selected_mask)
        n_selected = selected_mask.sum()
        print(f"\nPseudo-labeling summary  [method={self.threshold_method}]")
        print(f"  Unlabeled samples : {n_total:,}")
        print(
            f"  Selected          : {n_selected:,}  ({100 * n_selected / max(n_total, 1):.1f}%)"
        )
        print(f"  Rejected          : {n_total - n_selected:,}")
        print()

        header = (
            f"  {'Class':<12} {'Threshold':>10} {'Candidates':>12} "
            f"{'Selected':>10} {'Pct':>8} {'Mean conf (sel)':>16}"
        )
        print(header)
        print("  " + "-" * (len(header) - 2))

        for cls in self.classes_:
            candidates_mask = pseudo_labels == cls
            sel_mask = candidates_mask & selected_mask
            n_cand = candidates_mask.sum()
            n_sel = sel_mask.sum()
            pct = 100 * n_sel / max(n_cand, 1)
            mean_conf = confidences[sel_mask].mean() if n_sel > 0 else float("nan")
            thresh = self.thresholds_.get(cls, float("nan"))
            print(
                f"  {str(cls):<12} {thresh:>10.4f} {n_cand:>12,} {n_sel:>10,} "
                f"{pct:>7.1f}% {mean_conf:>16.4f}"
            )
        print()


if __name__ == "__main__":
    import numpy as np
    import pandas as pd
    from lightgbm import LGBMClassifier
    from sklearn.datasets import make_classification
    from sklearn.model_selection import StratifiedKFold
    from sklearn.pipeline import Pipeline

    from kvbiii_ml.data_processing.preprocessing.outlier_handling.winsorizer_trimmer import (
        WinsorizerWithOriginal,
    )

    RANDOM_STATE = 42
    N_LABELED = 2_000
    N_UNLABELED = 5_000
    N_FEATURES = 10
    N_FOLDS = 3
    ES = 30
    FEATURE_NAMES = [f"feature_{i}" for i in range(N_FEATURES)]

    def _make_data(n: int, n_classes: int, seed: int) -> tuple[pd.DataFrame, pd.Series]:
        """Generate synthetic classification dataset with deliberate noise for realism."""
        x_arr, y_arr = make_classification(
            n_samples=n,
            n_features=N_FEATURES,
            n_informative=4,
            n_redundant=4,
            n_classes=n_classes,
            n_clusters_per_class=1,
            flip_y=0.05,
            random_state=seed,
        )
        return pd.DataFrame(x_arr, columns=FEATURE_NAMES), pd.Series(
            y_arr, name="target"
        )

    def _build_pipeline() -> Pipeline:
        """Build a standard preprocessing pipeline."""
        return Pipeline(
            [
                (
                    "winsorizer",
                    WinsorizerWithOriginal(
                        variables=FEATURE_NAMES, capping_method="gaussian", tail="both"
                    ),
                )
            ]
        )

    def _build_cv(
        metric: str, pipeline: Pipeline | None = None
    ) -> CrossValidationTrainer:
        """Build a cross-validator for the given metric."""
        return CrossValidationTrainer(
            metric_name=metric,
            problem_type="classification",
            cv=StratifiedKFold(
                n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE
            ),
            preprocessing_pipeline=pipeline,
            verbose=False,
        )

    def _run_scenario(
        label: str,
        estimator: BaseEstimator,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        x_unlabeled: pd.DataFrame,
        threshold_method: str,
        metric: str,
        pipeline: Pipeline | None = None,
        **kwargs,
    ) -> None:
        """Run one pseudo-labeling scenario and report augmentation results."""
        print("=" * 75)
        print(f"Scenario: {label}")
        print("=" * 75)

        cv = _build_cv(metric, pipeline)
        gen = PseudoLabelGenerator(
            cross_validator=cv, threshold_method=threshold_method, **kwargs
        )
        gen.fit(estimator, X_train, y_train, x_unlabeled)
        x_aug, y_aug, _w_aug = gen.get_augmented_dataset()

        print(f"  Original labeled    : {len(X_train):,}")
        print(f"  Pseudo-labeled      : {len(x_aug) - len(X_train):,}")
        print(f"  Total for retraining: {len(x_aug):,}")
        print()

        cv_retrain = _build_cv(metric, pipeline)
        lgbm_retrain = LGBMClassifier(
            n_estimators=200,
            early_stopping_rounds=ES,
            verbose=-1,
            random_state=RANDOM_STATE,
        )
        _, valid_aug, _ = cv_retrain.fit(lgbm_retrain, x_aug, y_aug)
        print(
            f"  Baseline CV valid   : {np.mean(cv.valid_scores_):.4f} ± {np.std(cv.valid_scores_):.4f}"
        )
        print(
            f"  Augmented CV valid  : {np.mean(valid_aug):.4f} ± {np.std(valid_aug):.4f}"
        )
        print()
        gen.plot_pseudo_label_stats()

    def _run_demo() -> None:
        """Run the pseudo-labeling scenario(s) on synthetic data."""
        lgbm = LGBMClassifier(
            n_estimators=200,
            early_stopping_rounds=ES,
            verbose=-1,
            random_state=RANDOM_STATE,
        )

        # print("\n>>> Scenario 1: Binary + auto threshold + preprocessing pipeline")
        x_tr, y_tr = _make_data(N_LABELED, n_classes=2, seed=RANDOM_STATE)
        x_unl, _ = _make_data(N_UNLABELED, n_classes=2, seed=RANDOM_STATE + 99)
        # _run_scenario(
        #     "Binary + LightGBM + auto threshold + pipeline",
        #     lgbm, x_tr, y_tr, x_unl,
        #     threshold_method="auto", metric="Log Loss",
        #     pipeline=_build_pipeline(), threshold_percentile=80.0,
        # )

        print("\n>>> Scenario 2: Binary + top_k threshold (top 20%)")
        _run_scenario(
            "Binary + LightGBM + top_k threshold",
            lgbm,
            x_tr,
            y_tr,
            x_unl,
            threshold_method="top_k",
            metric="Log Loss",
            top_k_pct=0.20,
        )

        # print("\n>>> Scenario 3: Multiclass + auto threshold")
        # x_tr_mc, y_tr_mc = _make_data(N_LABELED, n_classes=3, seed=RANDOM_STATE)
        # x_unl_mc, _ = _make_data(N_UNLABELED, n_classes=3, seed=RANDOM_STATE + 99)
        # _run_scenario(
        #     "Multiclass (3 classes) + LightGBM + auto threshold",
        #     lgbm,
        #     x_tr_mc,
        #     y_tr_mc,
        #     x_unl_mc,
        #     threshold_method="auto",
        #     metric="Log Loss",
        #     threshold_percentile=75.0,
        # )

    _run_demo()
