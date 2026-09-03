import warnings
from typing import ClassVar

import numpy as np
import optuna
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.model_selection import KFold

from kvbiii_ml.modeling.training.cross_validation import CrossValidationTrainer

warnings.filterwarnings(
    "ignore",
    category=optuna.exceptions.ExperimentalWarning,
    message=".*multivariate.*",
)


class EnsembleWeightTunerCV:
    """Tune ensemble weights by optimizing a validation metric.

    This tuner fits each estimator on cross-validation folds, caches their
    validation predictions, and uses Optuna to find weights that best blend
    these predictions according to the chosen metric.

    Every candidate weight vector, including the one Optuna reports as best,
    is scored as the mean metric across CV folds minus a stability penalty on
    its cross-fold standard deviation, rather than on the pooled OOF set.
    This mirrors how the base estimators are already validated elsewhere in
    this codebase (mean +/- std per fold) and keeps the search from
    rewarding a weight vector that only exploits noise in one fixed pooled
    sample. After the search, the winning weights are chosen by comparing
    the search result against uniform averaging and against each single
    estimator on that same robust score, so tuning can never hand back
    weights that score worse than simply averaging the estimators on the
    evidence it was given.
    """

    MIN_WEIGHT_MASS: ClassVar[float] = 1e-6
    FOLD_STD_PENALTY: ClassVar[float] = 0.5

    def __init__(
        self,
        estimators: list[BaseEstimator],
        cross_validator: CrossValidationTrainer,
        n_trials: int = 50,
        seed: int = 17,
        allow_negative_weights: bool = False,
    ) -> None:
        """Initialize the tuner.

        Args:
            estimators (list[BaseEstimator]): Base estimators to blend.
            cross_validator (CrossValidationTrainer): Cross-validation trainer to use.
            n_trials (int): Number of Optuna trials. Defaults to 50.
            seed (int): Random seed for the TPE sampler. Defaults to 17.
            allow_negative_weights (bool): Whether to allow negative blending weights. Defaults to False.
        """
        self.estimators = list(estimators)
        self.cross_validator = cross_validator
        self.n_trials = n_trials
        self.seed = seed
        self.allow_negative_weights = allow_negative_weights
        self.fold_std_penalty = self.FOLD_STD_PENALTY
        self.problem_type = cross_validator.problem_type
        self.metric_fn = cross_validator.metric_fn
        self.metric_type = cross_validator.metric_type
        self.metric_direction = cross_validator.metric_direction
        self.best_weights: np.ndarray | None = None
        self.best_score_: float | None = None
        self.selection_source_: str | None = None

    def tune(self, X: pd.DataFrame, y: pd.Series) -> optuna.study.Study:
        """Run Optuna to find blending weights that beat naive baselines on cross-validated evidence.

        Args:
            X (pd.DataFrame): Feature matrix.
            y (pd.Series): Target vector.

        Returns:
            optuna.study.Study: Completed study. Best weights are stored in
                ``self.best_weights``; ``self.best_score_`` and
                ``self.selection_source_`` record what was selected and why.
        """
        X, y = self.check_x(X), self.check_y(y)
        y_true, preds_list, fold_boundaries = self._perform_cv(X, y)
        candidates = self._baseline_candidates(len(self.estimators))
        study = self._create_study()
        self._enqueue_baselines(study, candidates)
        study.optimize(
            lambda trial: self._objective(trial, y_true, preds_list, fold_boundaries),
            n_trials=self.n_trials,
        )
        self.best_weights, self.best_score_, self.selection_source_ = self._select_best(
            study, y_true, preds_list, fold_boundaries, candidates
        )
        if self.cross_validator.verbose:
            print(
                f"[EnsembleWeightTunerCV] Selected '{self.selection_source_}' weights "
                f"(score={self.best_score_:.5f}): {np.round(self.best_weights, 4)}"
            )
        return study

    @staticmethod
    def check_x(X: pd.DataFrame | np.ndarray | list | dict) -> pd.DataFrame:
        """Ensure feature input is a pandas DataFrame.

        Args:
            X (pd.DataFrame | np.ndarray | list | dict): Feature input to convert.

        Returns:
            pd.DataFrame: Features as a DataFrame.
        """
        return X if isinstance(X, pd.DataFrame) else pd.DataFrame(X)

    @staticmethod
    def check_y(y: pd.Series | np.ndarray | list) -> pd.Series:
        """Ensure target input is a pandas Series.

        Args:
            y (pd.Series | np.ndarray | list): Target input to convert.

        Returns:
            pd.Series: Target as a Series.
        """
        return y if isinstance(y, pd.Series) else pd.Series(y)

    @staticmethod
    def _baseline_candidates(n_estimators: int) -> dict[str, np.ndarray]:
        """Builds safe baseline weight vectors to compare the search result against.

        Args:
            n_estimators (int): Number of estimators being blended.

        Returns:
            dict[str, np.ndarray]: Uniform blend plus, when more than one
                estimator is present, a one-hot vector selecting each
                individual estimator.
        """
        candidates = {"uniform": np.full(n_estimators, 1.0 / n_estimators)}
        if n_estimators > 1:
            for i in range(n_estimators):
                one_hot = np.zeros(n_estimators)
                one_hot[i] = 1.0
                candidates[f"single_estimator_{i}"] = one_hot
        return candidates

    def _blend_predictions(
        self, preds_list: list[pd.Series | pd.DataFrame], weights: np.ndarray
    ) -> pd.Series | pd.DataFrame:
        """Blend predictions with weights while preserving original indices.

        Whenever any weight is actually negative, uses logit (binary) or
        log-softmax (multiclass) blending to keep classification outputs in
        valid probability space. Otherwise uses a direct weighted sum, which
        is exact and avoids unnecessary logit round-tripping whenever the
        sampled weights happen to be non-negative even under
        ``allow_negative_weights=True``.

        Args:
            preds_list (list[pd.Series | pd.DataFrame]): Per-estimator OOF predictions.
            weights (np.ndarray): Normalized blending weights (sum to 1).

        Returns:
            pd.Series | pd.DataFrame: Blended predictions with original indices.
        """
        first = preds_list[0]
        is_df = isinstance(first, pd.DataFrame)
        has_negative = bool(np.any(weights < 0.0))
        neg_clf = self.problem_type == "classification" and has_negative
        stacked = np.stack([p.values for p in preds_list])

        if neg_clf and not is_df:
            eps = 1e-9
            logits = np.log(
                np.clip(stacked, eps, 1 - eps) / np.clip(1 - stacked, eps, 1 - eps)
            )
            blended = 1.0 / (1.0 + np.exp(-np.einsum("e,es->s", weights, logits)))
        elif neg_clf:
            eps = 1e-12
            logp = np.log(np.clip(stacked, eps, 1.0))
            scores = np.einsum("e,esc->sc", weights, logp)
            scores -= scores.max(axis=1, keepdims=True)
            exp_s = np.exp(scores)
            blended = exp_s / np.clip(exp_s.sum(axis=1, keepdims=True), eps, None)
        else:
            blended = np.einsum("e,e...->...", weights, stacked)
            if is_df:
                row_sums = blended.sum(axis=1, keepdims=True)
                blended = np.clip(
                    blended / np.where(row_sums == 0.0, 1.0, row_sums), 0.0, 1.0
                )
                blended /= np.clip(blended.sum(axis=1, keepdims=True), 1e-12, None)

        if is_df:
            return pd.DataFrame(blended, index=first.index, columns=first.columns)
        return pd.Series(blended, index=first.index)

    def _collect_fold_predictions(
        self, X: pd.DataFrame, splits: list[tuple[np.ndarray, np.ndarray]]
    ) -> pd.Series | pd.DataFrame:
        """Applies each fold's fitted pipeline and estimator to that fold's validation split.

        Args:
            X (pd.DataFrame): Full feature matrix.
            splits (list[tuple[np.ndarray, np.ndarray]]): CV train/validation index pairs.

        Returns:
            pd.Series | pd.DataFrame: This estimator's pooled OOF predictions, in fold order.
        """
        est_preds = []
        for (_, val_idx), fitted_est, fold_pipeline in zip(
            splits,
            self.cross_validator.fitted_estimators_,
            self.cross_validator.fitted_pipelines_,
        ):
            X_valid = CrossValidationTrainer._transform_with_pipeline(
                fold_pipeline, X.iloc[val_idx]
            )
            if self.problem_type == "classification":
                proba = fitted_est.predict_proba(X_valid)
                if proba.shape[1] == 2:
                    pred = pd.Series(proba[:, 1], index=X.iloc[val_idx].index)
                else:
                    pred = pd.DataFrame(proba, index=X.iloc[val_idx].index)
            else:
                pred = pd.Series(
                    fitted_est.predict(X_valid), index=X.iloc[val_idx].index
                )
            est_preds.append(pred)
        return pd.concat(est_preds)

    def _create_study(self) -> optuna.study.Study:
        """Create an Optuna study with a TPE sampler and a pruner matched to per-fold reporting.

        MedianPruner (rather than HyperbandPruner) is used because trials
        only report one intermediate value per CV fold -- a handful of
        steps -- which is the regime MedianPruner targets, unlike Hyperband's
        many-step resource allocation designed for iterative training curves.

        Returns:
            optuna.study.Study: Configured study.
        """
        sampler = optuna.samplers.TPESampler(
            seed=self.seed, n_startup_trials=25, multivariate=True
        )
        pruner = optuna.pruners.MedianPruner(n_startup_trials=10, n_warmup_steps=1)
        return optuna.create_study(
            direction=self.metric_direction,
            sampler=sampler,
            pruner=pruner,
            study_name="Ensemble weight tuning",
        )

    @staticmethod
    def _enqueue_baselines(
        study: optuna.study.Study, candidates: dict[str, np.ndarray]
    ) -> None:
        """Seeds the study with baseline weight vectors as guaranteed first trials.

        Args:
            study (optuna.study.Study): Study to seed.
            candidates (dict[str, np.ndarray]): Baseline weight vectors from `_baseline_candidates`.
        """
        for weights in candidates.values():
            study.enqueue_trial({f"w{i}": float(w) for i, w in enumerate(weights)})

    def _normalize_weights(self, weights: np.ndarray) -> np.ndarray | None:
        """Rescales weights so they sum to exactly 1, preserving their sign.

        Dividing by the signed sum (rather than the L1 norm) keeps the
        blended prediction a genuine weighted average. This only affects
        rank-based metrics like Roc AUC when weights are negative, but it is
        required for any scale-sensitive metric (Log Loss, Brier Score, and
        every regression metric) to stay meaningful.

        Args:
            weights (np.ndarray): Raw weight vector suggested by Optuna.

        Returns:
            np.ndarray | None: Weights rescaled to sum to 1, or None when the
                signed sum is too close to zero to normalize safely.
        """
        total = weights.sum()
        if abs(total) <= self.MIN_WEIGHT_MASS:
            return None
        return weights / total

    def _objective(
        self,
        trial: optuna.Trial,
        y_true: pd.Series,
        preds_list: list[pd.Series | pd.DataFrame],
        fold_boundaries: list[int],
    ) -> float:
        """Objective function for weight tuning using cached OOF predictions.

        Args:
            trial (optuna.Trial): Current Optuna trial.
            y_true (pd.Series): OOF targets with preserved fold order.
            preds_list (list[pd.Series | pd.DataFrame]): Per-estimator OOF predictions.
            fold_boundaries (list[int]): Cumulative per-fold row counts for slicing.

        Returns:
            float: Mean-minus-std metric across folds for the sampled weight vector.

        Raises:
            optuna.TrialPruned: When the sampled weights are degenerate (near-zero
                signed sum) or when no fold yields a valid metric value.
        """
        n = len(self.estimators)
        lo, hi = (-1.0, 1.0) if self.allow_negative_weights else (0.0, 1.0)
        raw_weights = np.array(
            [trial.suggest_float(f"w{i}", lo, hi) for i in range(n)], dtype=float
        )
        weights = self._normalize_weights(raw_weights)
        if weights is None:
            raise optuna.TrialPruned(
                "Degenerate weight vector: weights summed too close to zero."
            )
        score = self._score_weights(
            weights, y_true, preds_list, fold_boundaries, trial=trial
        )
        if score is None:
            raise optuna.TrialPruned(
                "No cross-validation fold produced a valid metric for these weights."
            )
        return score

    def _perform_cv(
        self, X: pd.DataFrame, y: pd.Series
    ) -> tuple[pd.Series, list[pd.Series | pd.DataFrame], list[int]]:
        """Collect OOF predictions for each estimator across CV folds.

        Applies each fold's fitted pipeline to the validation split before calling
        the estimator, ensuring consistency with training-time transformations.
        Each estimator's own cross-validation fit is temporarily silenced so N
        estimators do not each reproduce the trainer's full per-iteration
        boosting logs; the trainer's original verbosity is restored afterwards
        and reused here to decide whether a one-line OOF summary is printed
        per estimator instead.

        Args:
            X (pd.DataFrame): Feature matrix.
            y (pd.Series): Target vector.

        Returns:
            tuple[pd.Series, list[pd.Series | pd.DataFrame], list[int]]: OOF
                true targets, per-estimator concatenated OOF predictions, and
                cumulative per-fold row counts usable as slice boundaries into
                both.
        """
        splits = list(self.cross_validator.cv.split(X, y))
        fold_boundaries = np.concatenate(
            ([0], np.cumsum([len(val_idx) for _, val_idx in splits]))
        ).tolist()
        y_valid_true = pd.concat([y.iloc[val_idx] for _, val_idx in splits])
        preds_per_estimator = []
        report_summary = bool(self.cross_validator.verbose)
        original_verbose = self.cross_validator.verbose

        for estimator in self.estimators:
            self.cross_validator.verbose = False
            try:
                self.cross_validator.fit(estimator, X, y)
            finally:
                self.cross_validator.verbose = original_verbose

            pooled_preds = self._collect_fold_predictions(X, splits)
            preds_per_estimator.append(pooled_preds)
            if report_summary:
                self._summarize_oof(
                    estimator, y_valid_true, pooled_preds, fold_boundaries
                )

        return y_valid_true, preds_per_estimator, fold_boundaries

    def _score_weights(
        self,
        weights: np.ndarray,
        y_true: pd.Series,
        preds_list: list[pd.Series | pd.DataFrame],
        fold_boundaries: list[int],
        trial: optuna.Trial | None = None,
    ) -> float | None:
        """Scores one weight vector as mean-minus-std across cross-validation folds.

        Evaluating per fold instead of on the pooled OOF set discourages
        weights that only exploit noise in a single pooled sample, matching
        how the base estimators themselves are already validated (mean +/-
        std per fold).

        Args:
            weights (np.ndarray): Normalized blending weights.
            y_true (pd.Series): OOF targets with preserved fold order.
            preds_list (list[pd.Series | pd.DataFrame]): Per-estimator OOF predictions.
            fold_boundaries (list[int]): Cumulative per-fold row counts for slicing.
            trial (optuna.Trial | None): Active trial for pruning/reporting. Defaults to None.

        Returns:
            float | None: Penalized aggregate score, or None if no fold produced a
                valid metric (e.g. a fold contains a single class for Roc AUC).

        Raises:
            optuna.TrialPruned: When ``trial`` determines the running score is
                worse than the median of prior trials at the same fold step.
        """
        blended = self._blend_predictions(preds_list, weights)
        if self.problem_type == "classification" and self.metric_type == "preds":
            blended = self._threshold_predictions(blended)

        fold_scores: list[float] = []
        for step, (start, end) in enumerate(
            zip(fold_boundaries[:-1], fold_boundaries[1:])
        ):
            fold_y_true = y_true.iloc[start:end]
            fold_pred = blended.iloc[start:end]
            try:
                fold_scores.append(float(self.metric_fn(fold_y_true, fold_pred)))
            except ValueError:
                continue
            if trial is not None:
                trial.report(float(np.mean(fold_scores)), step=step)
                if trial.should_prune():
                    raise optuna.TrialPruned()

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
        y_true: pd.Series,
        preds_list: list[pd.Series | pd.DataFrame],
        fold_boundaries: list[int],
        candidates: dict[str, np.ndarray],
    ) -> tuple[np.ndarray, float, str]:
        """Picks the best-scoring weight vector among the search result and safe baselines.

        Guarantees the tuner never returns weights that score worse, on the
        same cross-validated evidence, than uniform averaging or the single
        best estimator.

        Args:
            study (optuna.study.Study): Completed Optuna study.
            y_true (pd.Series): OOF targets with preserved fold order.
            preds_list (list[pd.Series | pd.DataFrame]): Per-estimator OOF predictions.
            fold_boundaries (list[int]): Cumulative per-fold row counts for slicing.
            candidates (dict[str, np.ndarray]): Baseline weight vectors to compare against.

        Returns:
            tuple[np.ndarray, float, str]: Winning weights, its score, and a label
                identifying whether the search result or a baseline was chosen.

        Raises:
            RuntimeError: If no weight vector, including every baseline,
                produced a valid metric on any fold.
        """
        n = len(self.estimators)
        results: dict[str, tuple[np.ndarray, float]] = {}

        completed_trials = [
            t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE
        ]
        if completed_trials:
            optimized_raw = np.array(
                [study.best_params[f"w{i}"] for i in range(n)], dtype=float
            )
            optimized_weights = self._normalize_weights(optimized_raw)
            if optimized_weights is not None:
                results["optimized"] = (optimized_weights, study.best_value)
        else:
            warnings.warn(
                "All Optuna trials were pruned; falling back to baseline weight vectors.",
                stacklevel=2,
            )

        for name, weights in candidates.items():
            score = self._score_weights(weights, y_true, preds_list, fold_boundaries)
            if score is not None:
                results[name] = (weights, score)

        if not results:
            raise RuntimeError(
                "Could not score any weight vector; every cross-validation fold "
                "failed to produce a valid metric."
            )

        best_name = (
            max(results, key=lambda k: results[k][1])
            if self.metric_direction == "maximize"
            else min(results, key=lambda k: results[k][1])
        )
        best_weights, best_score = results[best_name]
        return best_weights, best_score, best_name

    def _summarize_oof(
        self,
        estimator: BaseEstimator,
        y_true: pd.Series,
        preds: pd.Series | pd.DataFrame,
        fold_boundaries: list[int],
    ) -> None:
        """Prints one mean +/- std OOF metric line per estimator.

        Mirrors ``CrossValidationTrainer``'s own fold-summary logging so a
        single-model view and a tuner view of the same estimator read the
        same way, instead of the raw per-iteration boosting logs being
        reproduced once per estimator.

        Args:
            estimator (BaseEstimator): Estimator these OOF predictions belong to.
            y_true (pd.Series): OOF targets with preserved fold order.
            preds (pd.Series | pd.DataFrame): This estimator's pooled OOF predictions.
            fold_boundaries (list[int]): Cumulative per-fold row counts for slicing.
        """
        fold_pred = preds
        if self.problem_type == "classification" and self.metric_type == "preds":
            fold_pred = self._threshold_predictions(preds)

        fold_scores = []
        for start, end in zip(fold_boundaries[:-1], fold_boundaries[1:]):
            try:
                fold_scores.append(
                    float(
                        self.metric_fn(
                            y_true.iloc[start:end], fold_pred.iloc[start:end]
                        )
                    )
                )
            except ValueError:
                continue
        if not fold_scores:
            return

        scores = np.array(fold_scores)
        print(
            f"[EnsembleWeightTunerCV] {type(estimator).__name__} OOF "
            f"{self.cross_validator.metric_name}: {scores.mean():.5f} +- {scores.std():.5f}"
        )

    @staticmethod
    def _threshold_predictions(blended: pd.Series | pd.DataFrame) -> pd.Series:
        """Converts blended probabilities into hard predictions for 'preds'-type metrics.

        Args:
            blended (pd.Series | pd.DataFrame): Blended probabilities.

        Returns:
            pd.Series: Predicted class labels.
        """
        if isinstance(blended, pd.DataFrame):
            return blended.idxmax(axis=1)
        return (blended >= 0.5).astype(int)


if __name__ == "__main__":
    from lightgbm import LGBMClassifier, LGBMRegressor
    from sklearn.datasets import make_classification, make_regression
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
        """Run EnsembleWeightTunerCV for both classification and regression."""
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

        clf_estimators = [
            LGBMClassifier(
                n_estimators=100, num_leaves=15, verbose=-1, random_state=RANDOM_STATE
            ),
            LGBMClassifier(
                n_estimators=100, num_leaves=31, verbose=-1, random_state=RANDOM_STATE
            ),
            LGBMClassifier(
                n_estimators=100, num_leaves=63, verbose=-1, random_state=RANDOM_STATE
            ),
        ]
        cv_clf = CrossValidationTrainer(
            metric_name="Roc AUC",
            problem_type="classification",
            cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE),
            preprocessing_pipeline=preprocessing_pipeline,
            verbose=False,
        )
        tuner_clf = EnsembleWeightTunerCV(
            estimators=clf_estimators,
            cross_validator=cv_clf,
            n_trials=50,
            seed=RANDOM_STATE,
            allow_negative_weights=False,
        )
        clf_study = tuner_clf.tune(x_df, y_ser)
        print("Best trial value:", clf_study.best_value)
        print("Best weights:", tuner_clf.best_weights)
        print("Selected from:", tuner_clf.selection_source_)

        print("\n=== Regression ===")
        x_reg, y_reg = make_regression(
            n_samples=N_SAMPLES,
            n_features=N_FEATURES,
            n_informative=5,
            noise=0.1,
            random_state=RANDOM_STATE,
        )
        x_reg_df = pd.DataFrame(x_reg, columns=FEATURE_NAMES)
        y_reg_ser = pd.Series(y_reg)

        reg_estimators = [
            LGBMRegressor(
                n_estimators=100, num_leaves=15, verbose=-1, random_state=RANDOM_STATE
            ),
            LGBMRegressor(
                n_estimators=100, num_leaves=31, verbose=-1, random_state=RANDOM_STATE
            ),
            LGBMRegressor(
                n_estimators=100, num_leaves=63, verbose=-1, random_state=RANDOM_STATE
            ),
        ]
        cv_reg = CrossValidationTrainer(
            metric_name="RMSE",
            problem_type="regression",
            cv=KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE),
            preprocessing_pipeline=preprocessing_pipeline,
            verbose=False,
        )
        tuner_reg = EnsembleWeightTunerCV(
            estimators=reg_estimators,
            cross_validator=cv_reg,
            n_trials=30,
            seed=RANDOM_STATE,
            allow_negative_weights=False,
        )
        reg_study = tuner_reg.tune(x_reg_df, y_reg_ser)
        print("Best trial value:", reg_study.best_value)
        print("Best weights:", tuner_reg.best_weights)
        print("Selected from:", tuner_reg.selection_source_)

    _run_demo()
