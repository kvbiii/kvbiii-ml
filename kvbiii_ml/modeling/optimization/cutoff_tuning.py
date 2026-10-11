import warnings
from typing import ClassVar

import numpy as np
import optuna
import pandas as pd
from sklearn.base import BaseEstimator

from kvbiii_ml.modeling.training.cross_validation import CrossValidationTrainer

warnings.filterwarnings(
    "ignore",
    category=optuna.exceptions.ExperimentalWarning,
    message=".*multivariate.*",
)


class CutoffTunerCV:
    """Tune decision cutoffs for classification using cross-validation.

    Binary:     single cutoff in (0, 1); predict 1 if P(1) >= cutoff.
    Multiclass: one cutoff per class; predict argmax(P / cutoff),
                which biases toward classes with lower cutoffs.

    Every candidate cutoff vector, including the one Optuna reports as best, is
    scored as the mean metric across cross-validation folds minus a stability
    penalty on its cross-fold standard deviation, rather than on the pooled OOF
    set: a cutoff chosen by scanning the exact sample it is judged on can otherwise
    fit that one sample's noise rather than a genuine decision boundary. This
    mirrors how ``EnsembleWeightTunerCV`` scores candidate blending weights
    elsewhere in this package. After the search, the winning cutoffs are chosen by
    comparing the search result against the untouched default decision rule (0.5
    for binary, plain argmax for multiclass) on that same penalized score:
    whichever scores best wins outright, so tuning can never hand back cutoffs
    that score worse than not tuning at all.
    """

    FOLD_STD_PENALTY: ClassVar[float] = 0.5
    BINARY_CUTOFF_BOUNDS: ClassVar[tuple[float, float]] = (0.01, 0.99)
    MULTICLASS_CUTOFF_BOUNDS: ClassVar[tuple[float, float]] = (0.01, 1.0)

    def __init__(
        self,
        estimator: BaseEstimator,
        cross_validator: CrossValidationTrainer,
        n_trials: int = 100,
        seed: int = 17,
        verbose: bool = True,
    ) -> None:
        """Initialize the tuner.

        Args:
            estimator (BaseEstimator): Unfitted estimator to train during CV.
            cross_validator (CrossValidationTrainer): Configured CV trainer.
            n_trials (int): Number of Optuna trials. Defaults to 100.
            seed (int): Random seed for the TPE sampler. Defaults to 17.
            verbose (bool): Whether to print the tuner's own start, per-trial, and
                finish logs. Independent of ``cross_validator.verbose``, which this
                tuner temporarily forces off during the estimator's cross-validation
                fit regardless. Defaults to True.
        """
        self.estimator = estimator
        self.cross_validator = cross_validator
        self.n_trials = n_trials
        self.seed = seed
        self.verbose = verbose
        self.fold_std_penalty = self.FOLD_STD_PENALTY
        self.metric_fn = cross_validator.metric_fn
        self.metric_direction = cross_validator.metric_direction
        self.metric_name = cross_validator.metric_name
        self.fitted_estimators_: list[BaseEstimator] = []
        self.best_cutoffs: np.ndarray | None = None
        self.best_score_: float | None = None
        self.selection_source_: str | None = None
        self.baseline_score_: float | None = None
        self.optimized_score_: float | None = None
        self._is_binary: bool = True

    def tune(self, X: pd.DataFrame, y: pd.Series) -> optuna.study.Study:
        """Fit CV, cache OOF probabilities, then optimise cutoffs that beat the default rule.

        Args:
            X (pd.DataFrame): Feature matrix.
            y (pd.Series): Target vector.

        Returns:
            optuna.study.Study: Completed study. Best cutoffs are stored in
                ``self.best_cutoffs`` with one entry per class -- for binary
                targets this is ``[1 - cutoff, cutoff]`` rather than a bare
                scalar. ``self.best_score_`` and ``self.selection_source_``
                record what was selected and why.
        """
        X, y = self._check_x(X), self._check_y(y)
        previous_verbosity = optuna.logging.get_verbosity()
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        try:
            self._log_start()
            y_true, y_proba, fold_boundaries = self._perform_cv(X, y)
            default_cutoffs = self._default_cutoffs(y_proba)
            study = self._create_study()
            study.enqueue_trial(self._cutoffs_to_params(default_cutoffs))
            study.optimize(
                lambda trial: self._objective(
                    trial, y_true, y_proba, fold_boundaries
                ),
                n_trials=self.n_trials,
                callbacks=[self._log_trial] if self.verbose else None,
            )
            self.best_cutoffs, self.best_score_, self.selection_source_ = (
                self._select_best(
                    study, y_true, y_proba, fold_boundaries, default_cutoffs
                )
            )
            self._log_finish()
        finally:
            optuna.logging.set_verbosity(previous_verbosity)
        return study

    def predict(
        self, X: pd.DataFrame, estimator: BaseEstimator | None = None
    ) -> np.ndarray:
        """Apply tuned cutoffs to produce class predictions.

        When no estimator is provided, uses the first fold's fitted estimator with
        its corresponding fitted pipeline applied to X first. Pass an external
        estimator when you want to use a model trained on the full dataset -
        pipeline transformation is then your responsibility.

        Args:
            X (pd.DataFrame): Feature matrix.
            estimator (BaseEstimator | None): Fitted estimator to use. Defaults to the
                first fold estimator with its pipeline applied.

        Returns:
            np.ndarray: Predicted class labels.

        Raises:
            RuntimeError: If called before tune().
        """
        if self.best_cutoffs is None:
            raise RuntimeError("Call tune() before predict().")
        if estimator is not None:
            y_proba = estimator.predict_proba(X)
        else:
            fold_pipeline = self.cross_validator.fitted_pipelines_[0]
            x_proc = CrossValidationTrainer._transform_with_pipeline(fold_pipeline, X)
            y_proba = self.fitted_estimators_[0].predict_proba(x_proc)
        return self._apply_cutoffs(y_proba, self.best_cutoffs, self._is_binary)

    @staticmethod
    def _apply_cutoffs(
        y_proba: np.ndarray, cutoffs: np.ndarray, is_binary: bool
    ) -> np.ndarray:
        """Convert probabilities and cutoffs into class predictions.

        Args:
            y_proba (np.ndarray): (n_samples,) for binary or (n_samples, n_classes).
            cutoffs (np.ndarray): Per-class cutoff values. For binary targets this may
                be a single-element ``[cutoff]`` (the internal search representation)
                or a two-element ``[1 - cutoff, cutoff]`` (``best_cutoffs``'s public
                shape); the last element is always the positive-class threshold.
            is_binary (bool): Whether to use scalar binary thresholding.

        Returns:
            np.ndarray: Integer class predictions of shape (n_samples,).
        """
        if is_binary:
            return (y_proba >= cutoffs[-1]).astype(int)
        return np.argmax(y_proba / np.clip(cutoffs, 1e-9, None), axis=1)

    @staticmethod
    def _check_x(X: object) -> pd.DataFrame:
        """Ensure feature input is a DataFrame.

        Args:
            X (object): Feature input.

        Returns:
            pd.DataFrame: Features as a DataFrame.
        """
        return X if isinstance(X, pd.DataFrame) else pd.DataFrame(X)

    @staticmethod
    def _check_y(y: object) -> pd.Series:
        """Ensure target input is a Series.

        Args:
            y (object): Target input.

        Returns:
            pd.Series: Target as a Series.
        """
        return y if isinstance(y, pd.Series) else pd.Series(y)

    def _create_study(self) -> optuna.study.Study:
        """Create an Optuna study with a TPE sampler and no automatic pruning.

        Each trial only re-thresholds OOF probabilities already cached once in
        ``_perform_cv`` and evaluates a metric on them -- there is no iterative,
        per-step cost to cut short, unlike the training-curve scenarios pruners
        are designed for.

        Returns:
            optuna.study.Study: Configured study.
        """
        sampler = optuna.samplers.TPESampler(
            seed=self.seed, n_startup_trials=25, multivariate=True
        )
        return optuna.create_study(
            direction=self.metric_direction,
            sampler=sampler,
            pruner=optuna.pruners.NopPruner(),
            study_name="Cutoff tuning",
        )

    def _cutoffs_to_params(self, cutoffs: np.ndarray) -> dict[str, float]:
        """Converts a cutoff vector into Optuna's suggest-parameter naming scheme.

        Args:
            cutoffs (np.ndarray): Cutoff vector to convert.

        Returns:
            dict[str, float]: ``{"cutoff": value}`` for binary targets, or
                ``{"cutoff_0": ..., "cutoff_1": ..., ...}`` for multiclass targets.
        """
        if self._is_binary:
            return {"cutoff": float(cutoffs[0])}
        return {f"cutoff_{i}": float(cutoff) for i, cutoff in enumerate(cutoffs)}

    def _default_cutoffs(self, y_proba: np.ndarray) -> np.ndarray:
        """Builds the untouched default decision rule's cutoff vector.

        Args:
            y_proba (np.ndarray): OOF probabilities, used only to read the class count.

        Returns:
            np.ndarray: ``[0.5]`` for binary targets, or all-ones for multiclass
                targets, both of which reduce ``_apply_cutoffs`` to the untuned
                default decision rule (P(1) >= 0.5, plain argmax).
        """
        return np.array([0.5]) if self._is_binary else np.ones(y_proba.shape[1])

    def _expand_cutoffs(self, cutoffs: np.ndarray) -> np.ndarray:
        """Widens a binary cutoff into a per-class vector matching the multiclass shape.

        Search and scoring work with a single binary cutoff internally, but a
        public result of length 1 does not read as "one cutoff per class" the way
        the multiclass result does. Expanding it keeps ``best_cutoffs`` at length
        ``n_classes`` regardless of target cardinality.

        Args:
            cutoffs (np.ndarray): Internal cutoff vector; length 1 for binary
                targets, already length ``n_classes`` for multiclass targets.

        Returns:
            np.ndarray: ``[1 - cutoff, cutoff]`` for binary targets; unchanged
                for multiclass targets.
        """
        if self._is_binary:
            return np.array([1.0 - cutoffs[0], cutoffs[0]])
        return cutoffs

    def _explain_baseline_fallback(self) -> str:
        """Explains why the default cutoff was kept over the searched result.

        Returns:
            str: Human-readable reason, tailored to whether the search produced no
                usable candidate at all or simply scored lower than the default.
        """
        if self.optimized_score_ is None:
            return (
                f"No searched cutoff vector produced a valid, non-degenerate score, "
                f"so the default cutoff (score={self.baseline_score_:.5f}) was kept."
            )
        return (
            f"The search's best cutoffs scored {self.optimized_score_:.5f}, which "
            f"does not beat the default cutoff (score={self.baseline_score_:.5f}), "
            f"so the default cutoff was kept."
        )

    def _format_cutoffs(self, cutoffs: np.ndarray) -> dict[str, float]:
        """Labels a cutoff vector with each class's label for logging.

        Args:
            cutoffs (np.ndarray): Cutoff vector with one entry per class.

        Returns:
            dict[str, float]: Mapping from class label to the cutoff rounded to
                four decimals.
        """
        labels = (
            self.fitted_estimators_[0].classes_
            if self.fitted_estimators_
            else range(len(cutoffs))
        )
        return {str(label): round(float(cutoff), 4) for label, cutoff in zip(labels, cutoffs)}

    def _log_finish(self) -> None:
        """Prints the final cutoff-selection summary when verbose mode is enabled."""
        if not self.verbose:
            return
        print(
            f"\n🎯 Best {self.metric_name}: {self.best_score_:.5f} "
            f"(selected: '{self.selection_source_}')"
        )
        if self.selection_source_ == "optimized":
            print(f"✨ Search beat the default cutoff (scored {self.baseline_score_:.5f}).")
        else:
            print(f"🛑 {self._explain_baseline_fallback()}")
        print(f"🏆 Best cutoffs: {self._format_cutoffs(self.best_cutoffs)}")

    def _log_start(self) -> None:
        """Prints a search-start banner when verbose mode is enabled."""
        if not self.verbose:
            return
        print(
            f"🔍 Starting cutoff search with {self.n_trials} trials via "
            f"cross-validation, target metric: {self.metric_name} "
            f"({self.metric_direction}).\n"
            f"🎯 Tuning decision cutoffs for {type(self.estimator).__name__}."
        )

    def _log_trial(
        self, study: optuna.study.Study, trial: optuna.trial.FrozenTrial
    ) -> None:
        """Prints a one-line summary for a finished trial.

        Registered as an Optuna study callback, so it fires once per trial in
        place of Optuna's own default per-trial logging, which is muted for the
        duration of ``tune()`` in favor of formatting consistent with the rest of
        the package.

        Args:
            study (optuna.study.Study): The study being optimized.
            trial (optuna.trial.FrozenTrial): The trial that just finished.
        """
        duration = (
            (trial.datetime_complete - trial.datetime_start).total_seconds()
            if trial.datetime_complete and trial.datetime_start
            else float("nan")
        )
        prefix = f"Trial {trial.number + 1}/{self.n_trials}"
        if trial.state == optuna.trial.TrialState.PRUNED:
            print(f"✂️  {prefix} pruned ({duration:.2f}s)")
            return
        if trial.state != optuna.trial.TrialState.COMPLETE:
            print(f"⚠️  {prefix} {trial.state.name.lower()} ({duration:.2f}s)")
            return
        cutoffs = np.round(list(trial.params.values()), 4).tolist()
        marker = "🌟" if study.best_trial.number == trial.number else "🔁"
        print(
            f"{marker} {prefix} | {self.metric_name}: {trial.value:.6f} | "
            f"Best: {study.best_value:.6f} (trial {study.best_trial.number + 1}) | "
            f"{duration:.2f}s | cutoffs: {cutoffs}"
        )

    def _objective(
        self,
        trial: optuna.Trial,
        y_true: np.ndarray,
        y_proba: np.ndarray,
        fold_boundaries: list[int],
    ) -> float:
        """Objective function for cutoff optimisation using cached OOF probabilities.

        Args:
            trial (optuna.Trial): Current Optuna trial.
            y_true (np.ndarray): OOF true labels, in fold order.
            y_proba (np.ndarray): OOF probabilities, in fold order.
            fold_boundaries (list[int]): Cumulative per-fold row counts for slicing.

        Returns:
            float: Mean-minus-std metric across folds for the sampled cutoffs.

        Raises:
            optuna.TrialPruned: When no fold yields a valid metric value for the
                sampled cutoffs.
        """
        if self._is_binary:
            low, high = self.BINARY_CUTOFF_BOUNDS
            cutoffs = np.array([trial.suggest_float("cutoff", low, high)])
        else:
            low, high = self.MULTICLASS_CUTOFF_BOUNDS
            n_classes = y_proba.shape[1]
            cutoffs = np.array(
                [trial.suggest_float(f"cutoff_{i}", low, high) for i in range(n_classes)]
            )
        score = self._score_cutoffs(cutoffs, y_true, y_proba, fold_boundaries)
        if score is None:
            raise optuna.TrialPruned(
                "No cross-validation fold produced a valid metric for these cutoffs."
            )
        return score

    def _params_to_cutoffs(self, params: dict[str, float], n_classes: int) -> np.ndarray:
        """Converts Optuna's suggest-parameter naming scheme back into a cutoff vector.

        Args:
            params (dict[str, float]): Parameters keyed the way ``_objective`` sampled them.
            n_classes (int): Number of cutoffs expected; 1 for binary targets.

        Returns:
            np.ndarray: Cutoff vector in class order.
        """
        if self._is_binary:
            return np.array([params["cutoff"]], dtype=float)
        return np.array([params[f"cutoff_{i}"] for i in range(n_classes)], dtype=float)

    def _perform_cv(
        self, X: pd.DataFrame, y: pd.Series
    ) -> tuple[np.ndarray, np.ndarray, list[int]]:
        """Run CV and return concatenated OOF labels, probabilities, and fold boundaries.

        Applies each fold's fitted pipeline to the validation split before calling
        predict_proba, matching training-time transformations. The estimator's own
        cross-validation fit is temporarily silenced, regardless of the tuner's own
        ``verbose`` setting, so its per-fold logs are not reproduced here; the
        trainer's original verbosity is restored afterwards. Whether a one-line OOF
        summary at the default cutoff is printed instead is controlled by the
        tuner's own ``verbose`` flag.

        Args:
            X (pd.DataFrame): Feature matrix.
            y (pd.Series): Target vector.

        Returns:
            tuple[np.ndarray, np.ndarray, list[int]]: OOF true labels, OOF
                probabilities, and cumulative per-fold row counts usable as slice
                boundaries into both.
        """
        original_verbose = self.cross_validator.verbose
        self.cross_validator.verbose = False
        try:
            self.cross_validator.fit(self.estimator, X, y)
        finally:
            self.cross_validator.verbose = original_verbose
        self.fitted_estimators_ = self.cross_validator.fitted_estimators_

        splits = list(self.cross_validator.cv.split(X, y))
        fold_boundaries = np.concatenate(
            ([0], np.cumsum([len(val_idx) for _, val_idx in splits]))
        ).tolist()

        oof_true, oof_proba = [], []
        for (_, val_idx), est, fold_pipeline in zip(
            splits,
            self.cross_validator.fitted_estimators_,
            self.cross_validator.fitted_pipelines_,
        ):
            X_valid = CrossValidationTrainer._transform_with_pipeline(
                fold_pipeline, X.iloc[val_idx]
            )
            oof_proba.append(est.predict_proba(X_valid))
            oof_true.append(y.iloc[val_idx].to_numpy())

        y_true = np.concatenate(oof_true)
        y_proba_full = np.concatenate(oof_proba, axis=0)
        self._is_binary = y_proba_full.shape[1] == 2
        y_proba = y_proba_full[:, 1] if self._is_binary else y_proba_full

        if self.verbose:
            self._summarize_oof(y_true, y_proba, fold_boundaries)
        return y_true, y_proba, fold_boundaries

    def _score_cutoffs(
        self,
        cutoffs: np.ndarray,
        y_true: np.ndarray,
        y_proba: np.ndarray,
        fold_boundaries: list[int],
    ) -> float | None:
        """Scores one cutoff vector as mean-minus-std across cross-validation folds.

        Evaluating per fold instead of on the pooled OOF set discourages cutoffs
        that only exploit noise in a single pooled sample, matching how
        ``EnsembleWeightTunerCV`` scores candidate blending weights.

        Args:
            cutoffs (np.ndarray): Candidate cutoff vector.
            y_true (np.ndarray): OOF true labels, in fold order.
            y_proba (np.ndarray): OOF probabilities, in fold order.
            fold_boundaries (list[int]): Cumulative per-fold row counts for slicing.

        Returns:
            float | None: Penalized aggregate score, or None if no fold produced a
                valid metric (e.g. a fold contains a single class for Roc AUC).
        """
        y_pred = self._apply_cutoffs(y_proba, cutoffs, self._is_binary)
        fold_scores: list[float] = []
        for start, end in zip(fold_boundaries[:-1], fold_boundaries[1:]):
            try:
                fold_scores.append(
                    float(self.metric_fn(y_true[start:end], y_pred[start:end]))
                )
            except ValueError:
                continue

        if not fold_scores:
            return None

        scores = np.array(fold_scores)
        penalty = self.fold_std_penalty * float(scores.std())
        mean_score = float(scores.mean())
        return (
            mean_score - penalty
            if self.metric_direction == "maximize"
            else mean_score + penalty
        )

    def _select_best(
        self,
        study: optuna.study.Study,
        y_true: np.ndarray,
        y_proba: np.ndarray,
        fold_boundaries: list[int],
        default_cutoffs: np.ndarray,
    ) -> tuple[np.ndarray, float, str]:
        """Picks the best-scoring cutoff vector between the search result and the default rule.

        Guarantees the tuner never returns cutoffs that score worse, on the same
        cross-validated evidence, than the untouched default decision rule: the
        search result and the default are scored with the same mean-minus-std
        penalized metric, and whichever scores best (highest to maximize, lowest to
        minimize) is returned outright.

        Args:
            study (optuna.study.Study): Completed Optuna study.
            y_true (np.ndarray): OOF true labels, in fold order.
            y_proba (np.ndarray): OOF probabilities, in fold order.
            fold_boundaries (list[int]): Cumulative per-fold row counts for slicing.
            default_cutoffs (np.ndarray): Untuned baseline cutoff vector.

        Returns:
            tuple[np.ndarray, float, str]: Winning cutoffs (one entry per class,
                even for binary targets; see ``_expand_cutoffs``), its penalized
                score, and a label identifying whether the search result
                ("optimized") or the default ("default") was chosen.

        Raises:
            RuntimeError: If even the default cutoff fails to produce a valid metric
                on any fold.

        Note:
            Also records ``baseline_score_`` and ``optimized_score_`` on ``self`` so
            ``_log_finish`` can explain the outcome.
        """
        baseline_score = self._score_cutoffs(
            default_cutoffs, y_true, y_proba, fold_boundaries
        )
        if baseline_score is None:
            raise RuntimeError(
                "Could not score the default cutoff; every cross-validation fold "
                "failed to produce a valid metric."
            )
        self.baseline_score_ = baseline_score
        self.optimized_score_ = None

        completed_trials = [
            t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE
        ]
        if not completed_trials:
            warnings.warn(
                "All Optuna trials were pruned; falling back to the default cutoff.",
                stacklevel=2,
            )
            return self._expand_cutoffs(default_cutoffs), baseline_score, "default"

        n_classes = 1 if self._is_binary else y_proba.shape[1]
        optimized_cutoffs = self._params_to_cutoffs(study.best_params, n_classes)
        optimized_score = self._score_cutoffs(
            optimized_cutoffs, y_true, y_proba, fold_boundaries
        )
        if optimized_score is None:
            return self._expand_cutoffs(default_cutoffs), baseline_score, "default"
        self.optimized_score_ = optimized_score

        beats_baseline = (
            optimized_score > baseline_score
            if self.metric_direction == "maximize"
            else optimized_score < baseline_score
        )
        if beats_baseline:
            return self._expand_cutoffs(optimized_cutoffs), optimized_score, "optimized"
        return self._expand_cutoffs(default_cutoffs), baseline_score, "default"

    def _summarize_oof(
        self, y_true: np.ndarray, y_proba: np.ndarray, fold_boundaries: list[int]
    ) -> None:
        """Prints the detected target cardinality and OOF metric at the default cutoff.

        Mirrors ``EnsembleWeightTunerCV``'s own OOF summary so a tuner-level view of
        the estimator reads the same way as the sibling tuners in this package.

        Args:
            y_true (np.ndarray): OOF true labels, in fold order.
            y_proba (np.ndarray): OOF probabilities, in fold order.
            fold_boundaries (list[int]): Cumulative per-fold row counts for slicing.
        """
        n_cutoffs = 1 if self._is_binary else y_proba.shape[1]
        mode = "binary" if self._is_binary else f"multiclass ({n_cutoffs} classes)"
        print(f"🎛️  Detected {mode} target; tuning {n_cutoffs} cutoff(s).")

        default_cutoffs = self._default_cutoffs(y_proba)
        y_pred = self._apply_cutoffs(y_proba, default_cutoffs, self._is_binary)
        fold_scores = []
        for start, end in zip(fold_boundaries[:-1], fold_boundaries[1:]):
            try:
                fold_scores.append(
                    float(self.metric_fn(y_true[start:end], y_pred[start:end]))
                )
            except ValueError:
                continue
        if not fold_scores:
            return

        scores = np.array(fold_scores)
        print(
            f"[CutoffTunerCV] {type(self.estimator).__name__} OOF {self.metric_name} "
            f"at default cutoff: {scores.mean():.5f} +- {scores.std():.5f}"
        )


if __name__ == "__main__":
    from lightgbm import LGBMClassifier
    from sklearn.datasets import make_classification
    from sklearn.model_selection import StratifiedKFold
    from sklearn.pipeline import Pipeline
    from kvbiii_ml.data_processing.preprocessing.outlier_handling.winsorizer_trimmer import (
        WinsorizerWithOriginal,
    )

    RANDOM_STATE = 17
    N_SAMPLES = 2_000
    N_FEATURES = 10
    FEATURE_NAMES = [f"feature_{i}" for i in range(N_FEATURES)]

    def _run_demo() -> None:
        """Run CutoffTunerCV for both binary and multiclass classification."""
        preprocessing_pipeline = Pipeline(
            [
                (
                    "winsorizer",
                    WinsorizerWithOriginal(
                        variables=FEATURE_NAMES,
                        capping_method="gaussian",
                        tail="right",
                    ),
                ),
            ]
        )

        print("=== Binary classification ===")
        x_arr, y_arr = make_classification(
            n_samples=N_SAMPLES,
            n_features=N_FEATURES,
            n_informative=5,
            n_redundant=2,
            random_state=RANDOM_STATE,
        )
        x_df = pd.DataFrame(x_arr, columns=FEATURE_NAMES)
        y_ser = pd.Series(y_arr)

        cv_bin = CrossValidationTrainer(
            metric_name="Balanced Accuracy",
            problem_type="classification",
            cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE),
            preprocessing_pipeline=preprocessing_pipeline,
            verbose=False,
        )
        tuner_bin = CutoffTunerCV(
            estimator=LGBMClassifier(
                n_estimators=100, verbose=-1, random_state=RANDOM_STATE
            ),
            cross_validator=cv_bin,
            n_trials=30,
            seed=RANDOM_STATE,
        )
        study_bin = tuner_bin.tune(x_df, y_ser)
        print("Best trial value:", study_bin.best_value)
        print("Best cutoffs:", tuner_bin.best_cutoffs)
        print("Selected from:", tuner_bin.selection_source_)

        print("\n=== Multiclass classification ===")
        x_arr_mc, y_arr_mc = make_classification(
            n_samples=N_SAMPLES,
            n_features=N_FEATURES,
            n_informative=5,
            n_redundant=2,
            n_classes=3,
            n_clusters_per_class=1,
            random_state=RANDOM_STATE,
        )
        x_df_mc = pd.DataFrame(x_arr_mc, columns=FEATURE_NAMES)
        y_ser_mc = pd.Series(y_arr_mc)

        cv_mc = CrossValidationTrainer(
            metric_name="Balanced Accuracy",
            problem_type="classification",
            cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE),
            preprocessing_pipeline=preprocessing_pipeline,
            verbose=False,
        )
        tuner_mc = CutoffTunerCV(
            estimator=LGBMClassifier(
                n_estimators=100, verbose=-1, random_state=RANDOM_STATE
            ),
            cross_validator=cv_mc,
            n_trials=30,
            seed=RANDOM_STATE,
        )
        study_mc = tuner_mc.tune(x_df_mc, y_ser_mc)
        print("Best trial value:", study_mc.best_value)
        print("Best cutoffs:", tuner_mc.best_cutoffs)
        print("Selected from:", tuner_mc.selection_source_)

    _run_demo()
