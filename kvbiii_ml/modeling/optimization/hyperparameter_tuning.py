import threading
import warnings
from collections import Counter
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from typing import Any, ClassVar

import numpy as np
import optuna
import pandas as pd
from sklearn.base import BaseEstimator, clone
from sklearn.model_selection import KFold, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline

from kvbiii_ml.modeling.training.base_trainer import BaseTrainer, to_metric_proba
from kvbiii_ml.modeling.training.cross_validation import CrossValidationTrainer
from kvbiii_ml.modeling.training.oof_model import _get_n_iterations

warnings.filterwarnings(
    "ignore",
    category=optuna.exceptions.ExperimentalWarning,
    message=".*(multivariate|group|constant_liar).*",
)

GridEntry = tuple[str, list[object]] | tuple[str, list[object], dict[str, Any]]
ParamsGrid = dict[str, GridEntry]

CATBOOST_PREFIX = "CatBoost"
CATBOOST_ITERATIONS_PARAM = "iterations"
DEFAULT_ITERATIONS_PARAM = "n_estimators"
EARLY_STOPPING_PARAM_NAMES = ("early_stopping_round", "early_stopping_rounds")
CATEGORICAL_PARAM_NAMES = ("cat_features", "categorical_feature")
ROW_SCALED_PARAM_NAMES = (
    "min_child_samples",
    "min_data_in_leaf",
    "min_child_weight",
    "min_sum_hessian_in_leaf",
    "min_data_per_group",
)
NUMERIC_PARAM_TYPES = ("int", "float")
BASELINE_LABEL = "baseline"
TUNED_LABEL = "tuned"
BYTES_PER_GB = float(1024**3)
MIN_ROWS_PER_EARLY_STOPPING_SPLIT = 2
MIN_FOLDS_FOR_MARGIN = 2
MIN_FOLDS_FOR_PRUNING = 3
MIN_STARTUP_TRIALS = 10
MIN_PRUNER_STARTUP_TRIALS = 5
STARTUP_TRIAL_SHARE = 5
CONFIRM_TRIAL_SHARE = 4
EXPLORE_SEED_OFFSET = 1000
WIDEN_FACTOR = 1.5
SATURATION_POSITION = 0.15
CENSORING_SHARE = 0.95
DEGENERATE_SPAN_SHARE = 0.05
MIN_CATEGORIES_TO_NARROW = 2
UNBOUNDED = (float("-inf"), float("inf"))
PARAM_HARD_LIMITS: dict[str, tuple[float, float]] = {
    "cat_smooth": (0.0, float("inf")),
    "colsample_bylevel": (0.05, 1.0),
    "colsample_bynode": (0.05, 1.0),
    "colsample_bytree": (0.05, 1.0),
    "depth": (1.0, 16.0),
    "gamma": (0.0, float("inf")),
    "l2_leaf_reg": (0.0, float("inf")),
    "learning_rate": (1e-4, 1.0),
    "max_delta_step": (0.0, float("inf")),
    "max_depth": (1.0, float("inf")),
    "min_child_samples": (1.0, float("inf")),
    "min_child_weight": (0.0, float("inf")),
    "min_data_in_leaf": (1.0, float("inf")),
    "num_leaves": (2.0, float("inf")),
    "one_hot_max_size": (0.0, float("inf")),
    "random_strength": (0.0, float("inf")),
    "reg_alpha": (0.0, float("inf")),
    "reg_lambda": (0.0, float("inf")),
    "scale_pos_weight": (0.0, float("inf")),
    "subsample": (0.05, 1.0),
}


@dataclass
class _FoldData:
    """One fully preprocessed cross-validation fold."""

    x_fit: pd.DataFrame
    y_fit: pd.Series
    x_es: pd.DataFrame | None
    y_es: pd.Series | None
    x_valid: pd.DataFrame
    y_valid: pd.Series


@dataclass
class _EvaluationPlan:
    """Everything needed to score one configuration at a single fidelity level."""

    X: pd.DataFrame
    y: pd.Series
    splits: list[tuple[np.ndarray, np.ndarray]]
    pipeline: Pipeline | None
    label: str
    refit_rows: int
    cached: list[_FoldData] | None = None
    fidelity: float = 1.0

    @property
    def train_rows(self) -> float:
        """Average row count of a training fold, before the early-stopping carve-out."""
        return float(np.mean([len(train_idx) for train_idx, _ in self.splits]))


@dataclass
class _Candidate:
    """A scored configuration competing for best_params_."""

    params: dict[str, Any]
    score: float
    iterations: list[int]
    source: str


class _OrderedTrialLogger:
    """Study callback that prints finished trials in trial-number order.

    Parallel trials finish in whatever order their training happens to take, so a
    finished trial is buffered until every lower-numbered trial has been printed. The
    running best on each line covers only the trials printed before it, so the log
    reads exactly as a sequential run would.
    """

    def __init__(
        self, label: str, total: int, metric_name: str, maximize: bool
    ) -> None:
        """Initialize the logger.

        Args:
            label (str): Stage name shown in front of every line.
            total (int): Trial budget of the stage.
            metric_name (str): Metric name shown on every completed line.
            maximize (bool): Whether a larger metric value is better.
        """
        self.label = label
        self.total = total
        self.metric_name = metric_name
        self.maximize = maximize
        self._lock = threading.Lock()
        self._pending: dict[int, optuna.trial.FrozenTrial] = {}
        self._next_number = 0
        self._best: optuna.trial.FrozenTrial | None = None

    def __call__(
        self, _study: optuna.study.Study, trial: optuna.trial.FrozenTrial
    ) -> None:
        """Buffer a finished trial and print every trial that is now next in line.

        Args:
            _study (optuna.study.Study): The study being optimized, unused.
            trial (optuna.trial.FrozenTrial): The trial that just finished.
        """
        with self._lock:
            self._pending[trial.number] = trial
            while self._next_number in self._pending:
                self._print(self._pending.pop(self._next_number))
                self._next_number += 1

    def _print(self, trial: optuna.trial.FrozenTrial) -> None:
        """Print one trial line and advance the running best."""
        start, end = trial.datetime_start, trial.datetime_complete
        duration = (end - start).total_seconds() if start and end else float("nan")
        prefix = f"[{self.label}] Trial {trial.number + 1}/{self.total}"
        if trial.state != optuna.trial.TrialState.COMPLETE:
            icon = "✂️ " if trial.state == optuna.trial.TrialState.PRUNED else "⚠️ "
            print(f"{icon} {prefix} {trial.state.name.lower()} ({duration:.2f}s)")
            return
        improved = self._best is None or (
            trial.value > self._best.value
            if self.maximize
            else trial.value < self._best.value
        )
        if improved:
            self._best = trial
        print(
            f"{'🌟' if improved else '🔁'} {prefix} | {self.metric_name}: {trial.value:.6f} | "
            f"Best: {self._best.value:.6f} (trial {self._best.number + 1}) | "
            f"{duration:.2f}s | params: {trial.params}"
        )


class _FoldMedianPruner(optuna.pruners.BasePruner):
    """Prunes a trial whose running fold mean is worse than the median of completed trials.

    Optuna's MedianPruner compares the *best* value a trial reported over all steps, which
    suits learning curves but not cross-validation: a step here is a running mean over
    different folds, so an easy first fold would shield every trial from pruning. This
    pruner compares the running mean at the current step only, against the running means
    completed trials reported on exactly the same folds. The last fold is never pruned,
    because its evaluation has already been paid for.
    """

    def __init__(
        self, n_folds: int, n_startup_trials: int, n_warmup_steps: int = 1
    ) -> None:
        """Initialize the pruner.

        Args:
            n_folds (int): Folds each trial reports, one step per fold.
            n_startup_trials (int): Completed trials required before pruning starts.
            n_warmup_steps (int): Steps a trial always survives. Defaults to 1.
        """
        self.n_folds = n_folds
        self.n_startup_trials = n_startup_trials
        self.n_warmup_steps = n_warmup_steps

    def prune(self, study: optuna.study.Study, trial: optuna.trial.FrozenTrial) -> bool:
        """Decide whether the trial is worse than the median at its current fold.

        Args:
            study (optuna.study.Study): Study the trial belongs to.
            trial (optuna.trial.FrozenTrial): Trial being evaluated.

        Returns:
            bool: True when the trial should stop.
        """
        step = trial.last_step
        if step is None or step < self.n_warmup_steps or step >= self.n_folds - 1:
            return False
        completed = study.get_trials(
            deepcopy=False, states=(optuna.trial.TrialState.COMPLETE,)
        )
        if len(completed) < self.n_startup_trials:
            return False
        others = [
            t.intermediate_values[step]
            for t in completed
            if step in t.intermediate_values
        ]
        if not others:
            return False
        median = float(np.median(others))
        value = trial.intermediate_values[step]
        if study.direction == optuna.study.StudyDirection.MAXIMIZE:
            return value < median
        return value > median


class RandomSearchCV:
    """Two-stage hyperparameter tuner that never returns params worse than the defaults.

    An *explore* stage ranks many configurations cheaply on a fixed stratified
    subsample with fold-wise pruning, and a *confirm* stage searches a box narrowed
    around the explore elite at full fidelity on the caller's own splitter. The
    decision is always taken at full fidelity, so the explore stage can only cost a
    candidate its shortlist slot, never select a bad winner. Datasets too small to
    benefit from subsampling skip the explore stage and run a single full search.

    Three properties distinguish this tuner from a plain Optuna loop:

    - **No early-stopping leak.** Each training fold gives up ``es_holdout_size`` of
      its rows as a dedicated early-stopping split, so boosting never stops on the
      rows it is scored on.
    - **Preprocessing is fitted once per fold, not once per trial.** A fold's
      preprocessed matrices depend only on the fold and the pipeline, so they are
      materialised up front and reused by every trial.
    - **A baseline guard.** The estimator's own parameters are scored on the same
      folds, and a searched configuration is returned only when it beats them by more
      than one standard error of the paired per-fold differences.

    ``best_params_`` is the authoritative result. When the baseline wins, it holds the
    estimator's own parameters while ``study.best_params`` still holds the best
    searched configuration; ``selection_source_`` records which one was chosen.
    """

    FOLD_STD_PENALTY: ClassVar[float] = 0.5
    EXPLORE_N_SPLITS: ClassVar[int] = 5
    CACHE_MEMORY_BUDGET_GB: ClassVar[float] = 8.0

    def __init__(
        self,
        cross_validator: CrossValidationTrainer,
        n_trials: int = 100,
        seed: int = 17,
        verbose: bool = True,
        *,
        n_parallel_trials: int = 1,
        explore_fraction: float = 0.25,
        explore_min_rows: int = 100_000,
        es_holdout_size: float = 0.1,
        top_k: int = 15,
    ) -> None:
        """Initialize the tuner.

        Args:
            cross_validator (CrossValidationTrainer): Configured trainer supplying the
                metric, its direction, the splitter, and the preprocessing pipeline.
            n_trials (int): Trial budget of the explore stage, or of the single search
                when no explore stage runs. The confirm stage adds
                max(top_k, n_trials // 4) trials. Defaults to 100.
            seed (int): Random seed for the sampler and every internal split. Defaults to 17.
            verbose (bool): Print one line per trial plus start and finish reports.
                Optuna's own logging is muted during tune() either way. Defaults to True.
            n_parallel_trials (int): Trials evaluated concurrently as Optuna threads.
                Boosting libraries release the GIL while training, so keep
                n_parallel_trials times the estimator's own thread count within the
                physical cores. Above 1 the sampler uses a constant liar and completion
                order depends on timing, so a seed no longer reproduces the study
                exactly. Defaults to 1.
            explore_fraction (float): Row fraction of the explore subsample. The
                explore stage runs only when this fraction exceeds explore_min_rows,
                and 1.0 disables it. Defaults to 0.25.
            explore_min_rows (int): Row floor of the explore subsample. Defaults to 100_000.
            es_holdout_size (float): Fraction of each training fold carved out for early
                stopping; 0.0 disables the carve-out. Defaults to 0.1.
            top_k (int): Explore configurations that define the confirm-stage box and
                warm-start it. Defaults to 15.

        Raises:
            ValueError: If any staging option lies outside its valid range.
        """
        self.cross_validator = cross_validator
        self.n_trials = n_trials
        self.seed = seed
        self.verbose = verbose
        self.n_parallel_trials = n_parallel_trials
        self.explore_fraction = explore_fraction
        self.explore_min_rows = explore_min_rows
        self.es_holdout_size = es_holdout_size
        self.top_k = top_k
        self._validate_settings()
        self.metric_fn = cross_validator.metric_fn
        self.metric_type = cross_validator.metric_type
        self.metric_direction = cross_validator.metric_direction
        self.metric_name = cross_validator.metric_name
        self.problem_type = getattr(cross_validator, "problem_type", "classification")
        self.params_grid: ParamsGrid = {}
        self._reset_results()

    @staticmethod
    def check_x(X: object) -> pd.DataFrame:
        """Ensure features are a DataFrame.

        Args:
            X (object): Input features.

        Returns:
            pd.DataFrame: Features as a DataFrame.
        """
        return X if isinstance(X, pd.DataFrame) else pd.DataFrame(X)

    @staticmethod
    def check_y(y: object) -> pd.Series:
        """Ensure target is a Series.

        Args:
            y (object): Input target.

        Returns:
            pd.Series: Target as a Series.
        """
        return y if isinstance(y, pd.Series) else pd.Series(y)

    def create_study(
        self,
        pruner: optuna.pruners.BasePruner | None = None,
        n_startup_trials: int = 25,
    ) -> optuna.study.Study:
        """Create an Optuna study with a seeded multivariate TPE sampler.

        Args:
            pruner (optuna.pruners.BasePruner | None): Pruner to attach. Defaults to
                None, which disables pruning via NopPruner.
            n_startup_trials (int): Random trials before TPE takes over. Defaults to 25.

        Returns:
            optuna.study.Study: Configured study object.
        """
        sampler = optuna.samplers.TPESampler(
            seed=self.seed,
            n_startup_trials=n_startup_trials,
            multivariate=True,
            group=True,
            constant_liar=self.n_parallel_trials > 1,
        )
        return optuna.create_study(
            direction=self.metric_direction,
            sampler=sampler,
            pruner=pruner if pruner is not None else optuna.pruners.NopPruner(),
            study_name="Optuna tuning",
        )

    def evaluate_params(
        self,
        estimator: BaseEstimator,
        X: pd.DataFrame,
        y: pd.Series,
        params: dict[str, Any],
    ) -> tuple[float, list[float], list[int]]:
        """Score one configuration under exactly the protocol tune() uses.

        Lets a hand-tuned configuration be compared against a tuned one on equal terms:
        both early-stop on a split held out of the training fold.

        Args:
            estimator (BaseEstimator): Estimator template to clone per fold.
            X (pd.DataFrame): Training features.
            y (pd.Series): Training target.
            params (dict[str, Any]): Configuration to score.

        Returns:
            tuple[float, list[float], list[int]]: Penalised score, per-fold raw scores,
                and the boosting iterations each fold used.
        """
        X, y = self.check_x(X), self.check_y(y)
        plan = self._build_plan(X, y, self.cross_validator.cv, "evaluate")
        fold_scores, best_iterations = self._evaluate_config(plan, estimator, params)
        return self._penalized_score(fold_scores), fold_scores, best_iterations

    def get_param(
        self,
        trial: optuna.Trial,
        param_name: str,
        param_values: GridEntry,
    ) -> object:
        """Sample a single hyperparameter from the provided search space.

        Args:
            trial (optuna.Trial): Active Optuna trial.
            param_name (str): Name of the hyperparameter.
            param_values (GridEntry): Tuple of (type, values) with an optional third
                element carrying keyword options. Supported types: "int", "float",
                "categorical", "constant". Numeric types accept {"log": True} for
                geometric spacing and {"step": ...} for a lattice; Optuna forbids
                combining the two, so "step" is ignored on a log scale.

        Returns:
            object: Sampled value compatible with the estimator.

        Raises:
            ValueError: If param_type is not one of the supported types.
        """
        param_type, values, options = self._unpack_spec(param_values)
        if param_type == "constant":
            return values[0]
        if param_type == "categorical":
            return trial.suggest_categorical(param_name, values)
        if param_type not in NUMERIC_PARAM_TYPES:
            raise ValueError(f"Unknown param_type: {param_type}")
        low, high = values
        log = bool(options.get("log", False))
        if param_type == "int":
            step = 1 if log else options.get("step", 1)
            return trial.suggest_int(param_name, low, high, step=step, log=log)
        step = None if log else options.get("step")
        return trial.suggest_float(param_name, low, high, step=step, log=log)

    def objective(
        self,
        trial: optuna.Trial,
        plan: _EvaluationPlan,
        estimator: BaseEstimator,
        space: ParamsGrid | None = None,
    ) -> float:
        """Score one sampled configuration against a plan.

        Args:
            trial (optuna.Trial): Active Optuna trial.
            plan (_EvaluationPlan): Fidelity level to score the configuration at.
            estimator (BaseEstimator): Estimator template to clone per fold.
            space (ParamsGrid | None): Search space to sample from. Defaults to None,
                which uses the caller's full grid.

        Returns:
            float: Penalised cross-validated score for this trial.
        """
        grid = space if space is not None else self.params_grid
        params = {
            name: self.get_param(trial, name, spec) for name, spec in grid.items()
        }
        fold_scores, best_iterations = self._evaluate_config(
            plan, estimator, params, trial
        )
        trial.set_user_attr("fold_scores", fold_scores)
        trial.set_user_attr("best_iterations", best_iterations)
        trial.set_user_attr("params", params)
        return self._penalized_score(fold_scores)

    def tune(
        self,
        estimator: BaseEstimator,
        X: pd.DataFrame,
        y: pd.Series,
        params_grid: ParamsGrid,
        X_valid: pd.DataFrame | None = None,
        y_valid: pd.Series | None = None,
    ) -> optuna.study.Study:
        """Run the search and select parameters that provably beat the estimator's own.

        Args:
            estimator (BaseEstimator): Estimator to tune.
            X (pd.DataFrame): Training features.
            y (pd.Series): Training target.
            params_grid (ParamsGrid): Parameter search space.
            X_valid (pd.DataFrame | None): Optional validation features. Together with
                y_valid, a single hold-out split replaces cross-validation and the
                search runs single-stage. Defaults to None.
            y_valid (pd.Series | None): Optional validation target. Defaults to None.

        Returns:
            optuna.study.Study: The final-stage study. ``self.best_params_`` holds the
                authoritative selection.
        """
        self.params_grid = params_grid
        self._reset_results()
        X, y = self.check_x(X), self.check_y(y)
        use_holdout = X_valid is not None and y_valid is not None
        previous_verbosity = optuna.logging.get_verbosity()
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        try:
            self._log_start(use_holdout)
            if use_holdout:
                plan = self._build_holdout_plan(
                    X, y, self.check_x(X_valid), self.check_y(y_valid)
                )
                shortlist = []
            else:
                plan = self._build_plan(X, y, self.cross_validator.cv, "confirm")
                shortlist = self._run_explore_stage(estimator, X, y, plan)
            study = self._run_confirm_stage(plan, estimator, shortlist)
            if not shortlist:
                self.saturated_params_ = self._saturation_report(
                    self._top_params(study)
                )
            self._select_best(plan, estimator, study)
            self._log_finish(study)
        finally:
            optuna.logging.set_verbosity(previous_verbosity)
        return study

    def _baseline_params(
        self, estimator: BaseEstimator, snap: bool = True
    ) -> dict[str, Any]:
        """Build the estimator's own parameters for every grid entry it exposes.

        Parameters the estimator does not expose through get_params are omitted so the
        library default applies rather than an invented value.

        Args:
            estimator (BaseEstimator): Estimator template supplying current values.
            snap (bool): Whether values are clamped into the grid. Required to enqueue
                the baseline as a trial, but wrong when scoring it for the guard, which
                must compare against the estimator exactly as configured. Defaults to True.

        Returns:
            dict[str, Any]: Baseline configuration, possibly a subset of the grid.
        """
        current = estimator.get_params(deep=False)
        baseline: dict[str, Any] = {}
        for name, spec in self.params_grid.items():
            param_type, values, options = self._unpack_spec(spec)
            if param_type == "constant":
                baseline[name] = values[0]
            elif current.get(name) is not None:
                value = (
                    self._snap_to_space(param_type, values, options, current[name])
                    if snap
                    else current[name]
                )
                if value is not None:
                    baseline[name] = value
        return baseline

    def _baseline_trial(self, estimator: BaseEstimator) -> dict[str, Any] | None:
        """Return the snapped baseline as enqueueable trial params, or None if incomplete."""
        baseline = self._baseline_params(estimator)
        tunable = self._tunable_names()
        if not all(name in baseline for name in tunable):
            return None
        return {name: baseline[name] for name in tunable}

    def _beats(self, score: float, reference: float, margin: float = 0.0) -> bool:
        """Check whether a score improves on a reference by more than a margin.

        Args:
            score (float): Candidate score.
            reference (float): Score to beat.
            margin (float): Improvement that must be exceeded. Defaults to 0.0.

        Returns:
            bool: True when score is better than reference by more than margin.
        """
        if self.metric_direction == "maximize":
            return score > reference + margin
        return score < reference - margin

    @staticmethod
    def _bound_position(value: float, low: float, high: float, log: bool) -> float:
        """Place a value on a zero-to-one scale between two bounds, in log space if log."""
        if log and value > 0 and low > 0:
            return float((np.log(value) - np.log(low)) / (np.log(high) - np.log(low)))
        return float((value - low) / (high - low))

    def _build_fold(
        self,
        plan: _EvaluationPlan,
        fold_idx: int,
        train_idx: np.ndarray,
        valid_idx: np.ndarray,
    ) -> _FoldData:
        """Materialise one preprocessed fold, including its early-stopping split.

        The pipeline is cloned and fitted on the fit split alone; fit_transform is used
        because cross-fitting encoders produce different training output than fit
        followed by transform.

        Args:
            plan (_EvaluationPlan): Plan supplying the data and pipeline.
            fold_idx (int): Zero-based fold index, used to seed the split.
            train_idx (np.ndarray): Row positions of the training split.
            valid_idx (np.ndarray): Row positions of the validation split.

        Returns:
            _FoldData: Preprocessed matrices and targets for this fold.
        """
        fit_idx, es_idx = self._split_early_stopping(plan.y, train_idx, fold_idx)
        x_fit, y_fit = plan.X.iloc[fit_idx], plan.y.iloc[fit_idx]
        x_valid, y_valid = plan.X.iloc[valid_idx], plan.y.iloc[valid_idx]
        x_es = plan.X.iloc[es_idx] if es_idx is not None else None
        y_es = plan.y.iloc[es_idx] if es_idx is not None else None
        if plan.pipeline is not None:
            fitted = clone(plan.pipeline)
            x_fit = fitted.fit_transform(x_fit, y_fit)
            x_es = fitted.transform(x_es) if x_es is not None else None
            x_valid = fitted.transform(x_valid)
        if fold_idx == 0:
            self._remember_transformed_columns(x_fit)
        return _FoldData(
            x_fit=self._downcast(x_fit),
            y_fit=y_fit,
            x_es=self._downcast(x_es),
            y_es=y_es,
            x_valid=self._downcast(x_valid),
            y_valid=y_valid,
        )

    def _build_holdout_plan(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        X_valid: pd.DataFrame,
        y_valid: pd.Series,
    ) -> _EvaluationPlan:
        """Wrap a caller-supplied hold-out split as a cached single-fold plan.

        Args:
            X (pd.DataFrame): Training features.
            y (pd.Series): Training target.
            X_valid (pd.DataFrame): Validation features.
            y_valid (pd.Series): Validation target.

        Returns:
            _EvaluationPlan: Plan with exactly one fold covering the supplied split.
        """
        train_idx = np.arange(len(X))
        valid_idx = np.arange(len(X), len(X) + len(X_valid))
        plan = _EvaluationPlan(
            X=pd.concat([X, X_valid], axis=0, ignore_index=True),
            y=pd.concat([y, y_valid], axis=0, ignore_index=True),
            splits=[(train_idx, valid_idx)],
            pipeline=self.cross_validator.preprocessing_pipeline,
            label="holdout",
            refit_rows=len(X),
        )
        plan.cached = [self._build_fold(plan, 0, train_idx, valid_idx)]
        return plan

    def _build_plan(
        self, X: pd.DataFrame, y: pd.Series, cv: Any, label: str
    ) -> _EvaluationPlan:
        """Build an evaluation plan, caching its folds when they fit the memory budget.

        The first fold is always built so its footprint can be measured; the rest are
        materialised only when the extrapolated total stays within
        CACHE_MEMORY_BUDGET_GB, otherwise every trial recomputes them.

        Args:
            X (pd.DataFrame): Features for this fidelity level.
            y (pd.Series): Target for this fidelity level.
            cv (Any): Splitter providing train and validation positions.
            label (str): Stage name used in log lines.

        Returns:
            _EvaluationPlan: Plan with folds cached or left to be recomputed.
        """
        plan = _EvaluationPlan(
            X=X,
            y=y,
            splits=list(cv.split(X, y)),
            pipeline=self.cross_validator.preprocessing_pipeline,
            label=label,
            refit_rows=len(X),
        )
        first_fold = self._build_fold(plan, 0, *plan.splits[0])
        estimated_gb = self._fold_nbytes(first_fold) * len(plan.splits) / BYTES_PER_GB
        if estimated_gb > self.CACHE_MEMORY_BUDGET_GB:
            self._say(
                f"💾 Fold cache disabled for {label} stage: ~{estimated_gb:.2f} GB exceeds the "
                f"{self.CACHE_MEMORY_BUDGET_GB:.2f} GB budget; preprocessing runs per trial."
            )
            return plan
        remaining = enumerate(plan.splits[1:], start=1)
        plan.cached = [
            first_fold,
            *(self._build_fold(plan, idx, *split) for idx, split in remaining),
        ]
        self._say(
            f"💾 Cached {len(plan.cached)} preprocessed {label} folds (~{estimated_gb:.2f} GB), "
            f"reused across every trial."
        )
        return plan

    def _commit_selection(
        self, estimator: BaseEstimator, plan: _EvaluationPlan, candidate: _Candidate
    ) -> None:
        """Store the winning configuration in a form ready for a full-data refit.

        Early-stopping keys are dropped, because a refit on all rows has no eval_set to
        stop against, and replaced by a concrete boosting-round count.

        Args:
            estimator (BaseEstimator): Estimator template, used to pick the round-count key.
            plan (_EvaluationPlan): Full-fidelity plan the candidate was scored on.
            candidate (_Candidate): Winning configuration.
        """
        resolved = {
            name: value
            for name, value in candidate.params.items()
            if name not in EARLY_STOPPING_PARAM_NAMES
        }
        resolved.update(
            self._resolve_n_estimators(
                estimator,
                candidate.iterations,
                candidate.params,
                self._refit_scale(plan),
            )
        )
        self.best_params_ = self._reconcile_categorical_params(resolved)
        self.best_score_ = candidate.score
        self.best_iterations_ = candidate.iterations
        self.selection_source_ = candidate.source

    @staticmethod
    def _completed_trials(study: optuna.study.Study) -> list[optuna.trial.FrozenTrial]:
        """List the trials of a study that finished with a value."""
        return [
            trial
            for trial in study.trials
            if trial.state == optuna.trial.TrialState.COMPLETE
            and trial.value is not None
        ]

    @staticmethod
    def _dedupe(candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Drop duplicate configurations while preserving their order."""
        return list({tuple(sorted(c.items())): c for c in candidates}.values())

    @staticmethod
    def _downcast(frame: pd.DataFrame | None) -> pd.DataFrame | None:
        """Narrow float64 columns to float32 to halve cached fold memory."""
        if not isinstance(frame, pd.DataFrame):
            return frame
        float_columns = frame.select_dtypes(include=["float64"]).columns
        if len(float_columns) == 0:
            return frame
        return frame.astype(dict.fromkeys(float_columns, "float32"))

    def _evaluate_config(
        self,
        plan: _EvaluationPlan,
        estimator: BaseEstimator,
        params: dict[str, Any],
        trial: optuna.Trial | None = None,
    ) -> tuple[list[float], list[int]]:
        """Score one configuration across every fold of a plan.

        Early stopping runs on each fold's dedicated early-stopping split while the
        score is taken on the untouched validation split.

        Args:
            plan (_EvaluationPlan): Fidelity level to evaluate at.
            estimator (BaseEstimator): Estimator template to clone per fold.
            params (dict[str, Any]): Parameters in full-data units.
            trial (optuna.Trial | None): Trial receiving intermediate values for
                pruning. Defaults to None, which disables reporting.

        Returns:
            tuple[list[float], list[int]]: Per-fold validation scores and the boosting
                iteration counts each fold actually used.

        Raises:
            optuna.TrialPruned: When the pruner rejects the trial mid-evaluation.
        """
        fold_params = self._reconcile_categorical_params(
            self._scale_row_params(params, plan.fidelity)
        )
        predict = (
            (lambda fitted, x: to_metric_proba(BaseTrainer.predict_proba(fitted, x)))
            if self.metric_type == "probs"
            else BaseTrainer.predict
        )
        fold_scores: list[float] = []
        best_iterations: list[int] = []
        for step, fold in enumerate(self._iter_folds(plan)):
            fold_estimator = CrossValidationTrainer._clone_estimator(estimator)
            fold_estimator.set_params(**fold_params)
            fitted = BaseTrainer.fit_estimator(
                fold_estimator,
                fold.x_fit,
                fold.y_fit,
                fold.x_es,
                fold.y_es,
                verbose=False,
                eval_on_train=False,
            )
            x_valid = CrossValidationTrainer._order_x_for_estimator(
                fold.x_valid, fitted
            )
            fold_scores.append(
                float(self.metric_fn(fold.y_valid, predict(fitted, x_valid)))
            )
            n_iterations = _get_n_iterations(fitted)
            if n_iterations is not None:
                best_iterations.append(int(n_iterations))
            if trial is not None:
                trial.report(float(np.mean(fold_scores)), step)
                if trial.should_prune():
                    raise optuna.TrialPruned()
        return fold_scores, best_iterations

    @staticmethod
    def _expand_bound(name: str, bound: float, at_low: bool) -> float:
        """Push one grid bound outward by a single multiplicative widening step.

        Multiplicative expansion respects the parameter's scale (a max_depth floor of 3
        becomes 2, never a negative depth). Library limits cap the result, and a bound
        of zero is left alone because it has no multiplicative neighbour.

        Args:
            name (str): Parameter name, used to look up its library limits.
            bound (float): Current bound value.
            at_low (bool): Whether the low bound is being expanded.

        Returns:
            float: Expanded bound, clamped to what the library accepts.
        """
        hard_low, hard_high = PARAM_HARD_LIMITS.get(name, UNBOUNDED)
        if bound <= 0:
            return bound
        if at_low:
            return max(bound / WIDEN_FACTOR, hard_low)
        return min(bound * WIDEN_FACTOR, hard_high)

    @staticmethod
    def _explore_fidelity(
        explore_plan: _EvaluationPlan, confirm_plan: _EvaluationPlan
    ) -> float:
        """Explore training-fold rows as a fraction of confirm training-fold rows.

        Both the subsample size and the differing fold counts change how many rows a
        model sees, and row-count hyperparameters are rescaled by this ratio.

        Args:
            explore_plan (_EvaluationPlan): Plan of the explore stage.
            confirm_plan (_EvaluationPlan): Plan of the confirm stage.

        Returns:
            float: Ratio in (0, 1]; 1.0 when both stages train on as many rows.
        """
        if confirm_plan.train_rows <= 0:
            return 1.0
        return float(min(1.0, explore_plan.train_rows / confirm_plan.train_rows))

    @staticmethod
    def _fold_scores(trial: optuna.trial.FrozenTrial) -> list[float]:
        """Return the per-fold validation scores a trial recorded."""
        return list(trial.user_attrs.get("fold_scores", []))

    @staticmethod
    def _fold_nbytes(fold: _FoldData) -> float:
        """Measure the in-memory footprint of one materialised fold's feature matrices."""
        frames = (fold.x_fit, fold.x_es, fold.x_valid)
        return float(
            sum(
                frame.memory_usage(deep=False).sum()
                for frame in frames
                if frame is not None
            )
        )

    @staticmethod
    def _format_bounds(values: list[object]) -> str:
        """Render grid bounds or choices compactly for logging."""
        try:
            return "[" + ", ".join(f"{float(v):g}" for v in values) + "]"
        except (TypeError, ValueError):
            return str(list(values))

    @staticmethod
    def _is_censored(
        median_iterations: float,
        estimator: BaseEstimator,
        params: dict[str, Any],
        param_name: str,
    ) -> bool:
        """Check whether the folds hit the boosting-round ceiling instead of stopping early.

        Args:
            median_iterations (float): Median iteration count across folds.
            estimator (BaseEstimator): Estimator template, read when params is silent.
            params (dict[str, Any]): Winning configuration.
            param_name (str): Name of the round-count parameter for this family.

        Returns:
            bool: True when early stopping never fired on the typical fold.
        """
        ceiling = params.get(
            param_name, estimator.get_params(deep=False).get(param_name)
        )
        try:
            ceiling = float(ceiling)
        except (TypeError, ValueError):
            return False
        return ceiling > 0 and median_iterations >= CENSORING_SHARE * ceiling

    def _iter_folds(self, plan: _EvaluationPlan) -> Iterator[_FoldData]:
        """Yield each fold of a plan, from the cache when one was built."""
        if plan.cached is not None:
            yield from plan.cached
            return
        for fold_idx, (train_idx, valid_idx) in enumerate(plan.splits):
            yield self._build_fold(plan, fold_idx, train_idx, valid_idx)

    def _log_finish(self, study: optuna.study.Study) -> None:
        """Print the final selection summary when verbose mode is enabled."""
        if not self.verbose:
            return
        states = Counter(trial.state for trial in study.trials)
        print(
            f"\n🎯 Best {self.metric_name}: {self.best_score_:.6f} "
            f"(selected: {self.selection_source_})"
        )
        if self.selection_source_ == BASELINE_LABEL:
            searched = self._completed_trials(study)
            best_searched = study.best_value if searched else float("nan")
            print(
                f"🛑 Kept the estimator's own parameters: baseline {self.baseline_score_:.6f} vs "
                f"best searched {best_searched:.6f}. No searched configuration beat the baseline "
                f"by more than one standard error of their paired per-fold differences, so "
                f"best_params_ holds the baseline, not study.best_params."
            )
        print(f"🏆 Best params ({self.selection_source_}): {self.best_params_}")
        if self.best_iterations_:
            print(
                f"🌲 Per-fold best iterations: {self.best_iterations_} "
                f"(scaled for the full-data refit in best_params_)"
            )
        if self.n_estimators_censored_:
            print(
                "⚠️  Early stopping never fired: the folds ran into the boosting-round ceiling, "
                "so the returned round count extrapolates a censored number. Raise the ceiling "
                "or the learning-rate floor."
            )
        if self.saturated_params_:
            pressed = ", ".join(
                f"{name} ({side} bound {float(self.params_grid[name][1][side == 'high']):g})"
                for name, side in self.saturated_params_.items()
            )
            print(
                f"⚠️  The best configs pressed against grid bounds: {pressed}. Widen and re-run."
            )
        if self.confirm_ties_ > 1:
            print(
                f"⚖️  {self.confirm_ties_} trials tied within one standard error; the one with "
                f"the fewest boosting rounds was kept rather than the argmax of the noise."
            )
        print(
            f"✅ Completed: {states[optuna.trial.TrialState.COMPLETE]} | "
            f"✂️  Pruned: {states[optuna.trial.TrialState.PRUNED]} of {len(study.trials)} trials"
        )

    def _log_narrowing(self, narrowed: ParamsGrid) -> None:
        """Report the narrowed confirm-stage box and every bound the elite pressed against."""
        if not self.verbose:
            return
        changed = [
            (name, self.params_grid[name][1], spec[1])
            for name, spec in narrowed.items()
            if spec[0] != "constant"
            and list(spec[1]) != list(self.params_grid[name][1])
        ]
        if changed:
            print(
                f"🎯 Confirm stage searches a box narrowed around the top {self.top_k} explore configs:"
            )
            for name, before, after in changed:
                print(
                    f"    {name}: {self._format_bounds(before)} -> {self._format_bounds(after)}"
                )
        if self.saturated_params_:
            print(
                "⚠️  Grid bounds the elite configs pressed against (expanded by one step):"
            )
            for name, side in self.saturated_params_.items():
                bound = self.params_grid[name][1][0 if side == "low" else 1]
                print(f"    {name}: {side.upper()} bound {float(bound):g}")

    def _log_start(self, use_holdout: bool) -> None:
        """Print a search-start summary when verbose mode is enabled."""
        if not self.verbose:
            return
        mode = "hold-out validation" if use_holdout else "cross-validation"
        tunable = self._tunable_names()
        parallel = (
            f", {self.n_parallel_trials} in parallel"
            if self.n_parallel_trials > 1
            else ""
        )
        print(
            f"🔍 Starting random search with {self.n_trials} trials{parallel} via {mode}, "
            f"target metric: {self.metric_name} ({self.metric_direction}).\n"
            f"🎛️  Search space: {tunable} ({len(tunable)} tunable dimensions)"
        )

    def _make_explore_splitter(self) -> Any:
        """Build the explore-stage splitter, seeded apart from the caller's own.

        A different seed keeps the confirm stage independent evidence rather than a
        second reading of the folds the shortlist was chosen on.

        Returns:
            Any: Splitter with EXPLORE_N_SPLITS shuffled folds.
        """
        settings = {
            "n_splits": self.EXPLORE_N_SPLITS,
            "shuffle": True,
            "random_state": self.seed + EXPLORE_SEED_OFFSET,
        }
        try:
            return type(self.cross_validator.cv)(**settings)
        except TypeError:
            fallback = (
                StratifiedKFold if self.problem_type == "classification" else KFold
            )
            return fallback(**settings)

    def _make_pruner(self, n_folds: int) -> optuna.pruners.BasePruner:
        """Build a fold-wise median pruner, or disable pruning when folds are too few.

        Args:
            n_folds (int): Number of folds the plan will report.

        Returns:
            optuna.pruners.BasePruner: _FoldMedianPruner when pruning can help, else NopPruner.
        """
        if n_folds < MIN_FOLDS_FOR_PRUNING:
            return optuna.pruners.NopPruner()
        n_startup = max(MIN_PRUNER_STARTUP_TRIALS, self.n_trials // STARTUP_TRIAL_SHARE)
        return _FoldMedianPruner(n_folds=n_folds, n_startup_trials=n_startup)

    @staticmethod
    def _median_iterations(trial: optuna.trial.FrozenTrial) -> float:
        """Report a trial's median boosting-round count, or infinity when unknown."""
        iterations = trial.user_attrs.get("best_iterations") or []
        return float(np.median(iterations)) if iterations else float("inf")

    @staticmethod
    def _narrow_int(
        low: float,
        high: float,
        floor: float,
        ceiling: float,
        original: tuple[str, list[object], dict[str, Any]],
    ) -> tuple[str, list[object], dict[str, Any]]:
        """Round a narrowed range onto the integer lattice of the caller's grid entry.

        Args:
            low (float): Narrowed lower bound before rounding.
            high (float): Narrowed upper bound before rounding.
            floor (float): Lowest value the bound may take.
            ceiling (float): Highest value the bound may take.
            original (tuple[str, list[object], dict[str, Any]]): Entry to fall back to.

        Returns:
            tuple[str, list[object], dict[str, Any]]: Narrowed entry, or the original
                when no valid integer range survives.
        """
        param_type, values, options = original
        log = bool(options.get("log", False))
        int_low = max(int(np.floor(low)), int(np.ceil(floor)))
        int_high = min(int(np.ceil(high)), int(np.floor(ceiling)))
        if log:
            int_low = max(int_low, 1)
        step = int(options.get("step") or 1)
        if step > 1 and not log:
            anchor = int(values[0])
            int_low = anchor + int(np.ceil((int_low - anchor) / step)) * step
            int_high = int_low + ((int_high - int_low) // step) * step
        if int_high <= int_low:
            return original
        return param_type, [int_low, int_high], options

    def _narrow_numeric(
        self, name: str, spec: GridEntry, observed: list[Any]
    ) -> GridEntry:
        """Shrink one numeric grid entry around the values the elite actually chose.

        Args:
            name (str): Parameter name.
            spec (GridEntry): The caller's original grid entry.
            observed (list[Any]): Values this parameter took across the elite.

        Returns:
            GridEntry: Narrowed grid entry, or the original bounds when narrowing would
                not leave a valid range.
        """
        param_type, values, options = self._unpack_spec(spec)
        original = (param_type, list(values), options)
        grid_low, grid_high = float(values[0]), float(values[1])
        log = bool(options.get("log", False))
        elite_low, elite_high = float(min(observed)), float(max(observed))
        if log and elite_low > 0:
            low, high = elite_low / WIDEN_FACTOR, elite_high * WIDEN_FACTOR
        else:
            pad = max(
                (WIDEN_FACTOR - 1.0) * (elite_high - elite_low),
                DEGENERATE_SPAN_SHARE * (grid_high - grid_low),
            )
            low, high = elite_low - pad, elite_high + pad
        side = self.saturated_params_.get(name)
        floor = (
            self._expand_bound(name, grid_low, at_low=True)
            if side == "low"
            else grid_low
        )
        ceiling = (
            self._expand_bound(name, grid_high, at_low=False)
            if side == "high"
            else grid_high
        )
        low, high = max(low, floor), min(high, ceiling)
        if param_type == "int":
            return self._narrow_int(low, high, floor, ceiling, original)
        if log:
            low = max(low, float(np.nextafter(0.0, 1.0)))
        return (param_type, [low, high], options) if high > low else original

    def _narrow_space(self, elite: list[dict[str, Any]]) -> ParamsGrid:
        """Build the confirm-stage search box around the configurations the elite chose.

        Each numeric range shrinks to the elite's span widened by WIDEN_FACTOR, with a
        bound the elite pressed against pushed outward by one step, and each categorical
        keeps only the choices the elite used. Constants pass through unchanged.

        Args:
            elite (list[dict[str, Any]]): Best explore configurations, best first.

        Returns:
            ParamsGrid: Narrowed grid in the caller's format.
        """
        self.saturated_params_ = self._saturation_report(elite)
        narrowed: ParamsGrid = {}
        for name, spec in self.params_grid.items():
            param_type, values, options = self._unpack_spec(spec)
            observed = [config[name] for config in elite if name in config]
            if param_type == "constant" or not observed:
                narrowed[name] = spec
            elif param_type == "categorical":
                kept = [value for value in values if value in set(observed)]
                narrowed[name] = (
                    (param_type, kept, options)
                    if len(kept) >= MIN_CATEGORIES_TO_NARROW
                    else spec
                )
            else:
                narrowed[name] = self._narrow_numeric(name, spec, observed)
        self._log_narrowing(narrowed)
        return narrowed

    def _n_early_stopping_rows(self, n_train_rows: float) -> int:
        """Rows a training fold of this size gives up for early stopping; 0 when too few."""
        n_rows = int(n_train_rows * self.es_holdout_size)
        return n_rows if n_rows >= MIN_ROWS_PER_EARLY_STOPPING_SPLIT else 0

    def _n_startup_trials(self) -> int:
        """Random trials a full-grid search runs before TPE takes over."""
        return max(MIN_STARTUP_TRIALS, self.n_trials // STARTUP_TRIAL_SHARE)

    def _optimize(
        self,
        study: optuna.study.Study,
        objective: Callable[[optuna.Trial], float],
        n_trials: int,
        label: str,
    ) -> None:
        """Run one study with the configured concurrency and ordered trial logging.

        Args:
            study (optuna.study.Study): Study to optimize.
            objective (Callable[[optuna.Trial], float]): Trial objective.
            n_trials (int): Trial budget.
            label (str): Stage name shown on every trial line.
        """
        maximize = self.metric_direction == "maximize"
        logger = _OrderedTrialLogger(label, n_trials, self.metric_name, maximize)
        study.optimize(
            objective,
            n_trials=n_trials,
            n_jobs=self.n_parallel_trials,
            callbacks=[logger] if self.verbose else None,
        )

    def _penalized_score(self, fold_scores: list[float]) -> float:
        """Score a configuration as its fold mean penalised by FOLD_STD_PENALTY fold stds.

        Args:
            fold_scores (list[float]): Per-fold validation scores.

        Returns:
            float: Penalised score, oriented so the metric direction still applies.
        """
        maximize = self.metric_direction == "maximize"
        if not fold_scores:
            return float("-inf") if maximize else float("inf")
        scores = np.asarray(fold_scores, dtype=float)
        penalty = self.FOLD_STD_PENALTY * float(scores.std())
        return (
            float(scores.mean()) - penalty
            if maximize
            else float(scores.mean()) + penalty
        )

    def _reconcile_categorical_params(self, params: dict[str, Any]) -> dict[str, Any]:
        """Realign declared categorical columns with what the pipeline actually produced.

        A pipeline that encodes and then selects features replaces the raw categorical
        columns, so a cat_features list written against the raw frame names columns the
        estimator never sees, and CatBoost fails with an unhelpful ``list.index`` error.
        Surviving declarations are kept; when none survive, the frame's own non-numeric
        columns take their place.

        Args:
            params (dict[str, Any]): Parameters about to be applied to an estimator.

        Returns:
            dict[str, Any]: Copy of params with categorical declarations reconciled.
        """
        if not self._transformed_columns:
            return params
        available = set(self._transformed_columns)
        reconciled = dict(params)
        for name in CATEGORICAL_PARAM_NAMES:
            declared = params.get(name)
            if not isinstance(declared, (list, tuple)) or not declared:
                continue
            kept = [column for column in declared if column in available]
            if len(kept) == len(declared):
                continue
            reconciled[name] = kept or list(self._transformed_categoricals)
            missing = [column for column in declared if column not in available]
            self._warn_reconciled(name, missing, reconciled[name])
        return reconciled

    def _refit_scale(self, plan: _EvaluationPlan) -> float:
        """Ratio of the refit's row count to the rows each fold's model was fitted on."""
        fit_rows = plan.train_rows - self._n_early_stopping_rows(plan.train_rows)
        return plan.refit_rows / fit_rows if fit_rows > 0 else 1.0

    def _remember_transformed_columns(self, frame: pd.DataFrame) -> None:
        """Record the columns and non-numeric columns of a preprocessed training fold."""
        self._transformed_columns = list(frame.columns)
        self._transformed_categoricals = [
            column
            for column in frame.columns
            if not pd.api.types.is_numeric_dtype(frame[column])
        ]

    @staticmethod
    def _required_margin(
        fold_scores: list[float], reference_fold_scores: list[float]
    ) -> float:
        """Compute one standard error of the paired per-fold differences of two configurations.

        Both configurations are scored on the same folds, so how hard each fold is affects
        both alike and cancels in the per-fold difference. The spread of those differences
        is the noise separating the two; the spread of either configuration's raw fold
        scores is dominated by fold difficulty and would demand a far larger margin.

        Args:
            fold_scores (list[float]): Per-fold raw metric values of the challenger.
            reference_fold_scores (list[float]): Per-fold raw metric values it is compared to.

        Returns:
            float: Improvement the challenger must exceed; zero with fewer than two paired
                folds.
        """
        n_folds = len(fold_scores)
        if n_folds != len(reference_fold_scores) or n_folds < MIN_FOLDS_FOR_MARGIN:
            return 0.0
        differences = np.asarray(fold_scores, dtype=float) - np.asarray(
            reference_fold_scores
        )
        return float(np.std(differences, ddof=1) / np.sqrt(n_folds))

    def _reset_results(self) -> None:
        """Clear every fitted attribute so each tune() starts from a clean state."""
        self.best_params_: dict[str, Any] | None = None
        self.best_score_: float | None = None
        self.baseline_score_: float | None = None
        self.best_iterations_: list[int] = []
        self.selection_source_: str | None = None
        self.explore_study_: optuna.study.Study | None = None
        self.saturated_params_: dict[str, str] = {}
        self.confirm_ties_ = 0
        self.n_estimators_censored_ = False
        self._warned_categoricals: set[str] = set()
        self._transformed_columns: list[str] = []
        self._transformed_categoricals: list[str] = []

    def _resolve_n_estimators(
        self,
        estimator: BaseEstimator,
        best_iterations: list[int],
        params: dict[str, Any],
        refit_scale: float,
    ) -> dict[str, Any]:
        """Convert per-fold iteration counts into a boosting-round count for the refit.

        The median fold iteration count is scaled by how many more rows the refit sees
        than each fold's model did. When the folds ran into the round ceiling instead
        of stopping, the median is censored; that is flagged on n_estimators_censored_
        but the scaled value is still returned as the best estimate available.

        Args:
            estimator (BaseEstimator): Estimator template, used to pick the parameter name.
            best_iterations (list[int]): Iteration counts observed per fold.
            params (dict[str, Any]): Winning configuration, read for the round ceiling.
            refit_scale (float): Refit rows divided by rows fitted per fold.

        Returns:
            dict[str, Any]: Override to merge into the returned parameters; empty when
                no iteration count was observed.
        """
        if not best_iterations:
            return {}
        is_catboost = type(estimator).__name__.startswith(CATBOOST_PREFIX)
        param_name = (
            CATBOOST_ITERATIONS_PARAM if is_catboost else DEFAULT_ITERATIONS_PARAM
        )
        median_iterations = float(np.median(best_iterations))
        self.n_estimators_censored_ = self._is_censored(
            median_iterations, estimator, params, param_name
        )
        return {param_name: max(1, int(round(median_iterations * refit_scale)))}

    def _run_confirm_stage(
        self,
        plan: _EvaluationPlan,
        estimator: BaseEstimator,
        shortlist: list[dict[str, Any]],
    ) -> optuna.study.Study:
        """Run the full-fidelity search, warm-started from the explore shortlist if any.

        With a shortlist, the stage searches a box narrowed around it without pruning.
        Without one, it is the only search: the full grid with fold-wise pruning,
        warm-started from the estimator's own parameters.

        Args:
            plan (_EvaluationPlan): Full-fidelity evaluation plan.
            estimator (BaseEstimator): Estimator template to clone per fold.
            shortlist (list[dict[str, Any]]): Explore-stage configurations, best first.

        Returns:
            optuna.study.Study: Completed study.
        """
        candidates = self._dedupe(shortlist)
        if candidates:
            space = self._narrow_space(candidates)
            n_trials = max(self.top_k, self.n_trials // CONFIRM_TRIAL_SHARE)
            study = self.create_study(n_startup_trials=len(candidates))
            label = "confirm"
        else:
            space, n_trials, label = self.params_grid, self.n_trials, "search"
            study = self.create_study(
                pruner=self._make_pruner(len(plan.splits)),
                n_startup_trials=self._n_startup_trials(),
            )
            baseline = self._baseline_trial(estimator)
            candidates = [baseline] if baseline is not None else []
        for candidate in candidates:
            study.enqueue_trial(
                self._snap_candidate(candidate, space), skip_if_exists=True
            )
        self._optimize(
            study,
            lambda trial: self.objective(trial, plan, estimator, space),
            n_trials,
            label,
        )
        return study

    def _run_explore_stage(
        self,
        estimator: BaseEstimator,
        X: pd.DataFrame,
        y: pd.Series,
        confirm_plan: _EvaluationPlan,
    ) -> list[dict[str, Any]]:
        """Rank configurations cheaply on a fixed subsample and return the shortlist.

        Args:
            estimator (BaseEstimator): Estimator template to clone per fold.
            X (pd.DataFrame): Full training features.
            y (pd.Series): Full training target.
            confirm_plan (_EvaluationPlan): Full-fidelity plan, the fidelity reference.

        Returns:
            list[dict[str, Any]]: Up to top_k parameter dictionaries, best first; empty
                when the dataset is too small for an explore stage to pay off.
        """
        if not self._should_explore(len(X)):
            return []
        plan = self._build_plan(
            *self._subsample(X, y), self._make_explore_splitter(), "explore"
        )
        plan.fidelity = self._explore_fidelity(plan, confirm_plan)
        if plan.fidelity < 1.0:
            self._say(
                f"📐 Explore fidelity {plan.fidelity:.3f}: row-count parameters "
                f"({', '.join(ROW_SCALED_PARAM_NAMES[:2])}, ...) are scaled down while "
                f"training, so logged values stay in full-data units."
            )
        study = self.create_study(
            pruner=self._make_pruner(len(plan.splits)),
            n_startup_trials=self._n_startup_trials(),
        )
        baseline = self._baseline_trial(estimator)
        if baseline is not None:
            study.enqueue_trial(baseline, skip_if_exists=True)
        self._optimize(
            study,
            lambda trial: self.objective(trial, plan, estimator),
            self.n_trials,
            "explore",
        )
        self.explore_study_ = study
        return self._top_params(study)

    def _saturation_report(self, elite: list[dict[str, Any]]) -> dict[str, str]:
        """Find parameters whose elite configurations press against a grid bound.

        A search that keeps choosing values at the edge of its box is clipped by the box
        rather than by the data. A bound already at the library's own limit is never
        reported, because there is no room to expand into.

        Args:
            elite (list[dict[str, Any]]): Best explore configurations, best first.

        Returns:
            dict[str, str]: Parameter name mapped to the saturated side, "low" or "high".
        """
        report: dict[str, str] = {}
        for name, spec in self.params_grid.items():
            param_type, values, options = self._unpack_spec(spec)
            if param_type not in NUMERIC_PARAM_TYPES:
                continue
            observed = [float(config[name]) for config in elite if name in config]
            low, high = float(values[0]), float(values[1])
            if not observed or high <= low:
                continue
            log = bool(options.get("log", False))
            position = float(
                np.median([self._bound_position(v, low, high, log) for v in observed])
            )
            hard_low, hard_high = PARAM_HARD_LIMITS.get(name, UNBOUNDED)
            if position <= SATURATION_POSITION and low > hard_low:
                report[name] = "low"
            elif position >= 1.0 - SATURATION_POSITION and high < hard_high:
                report[name] = "high"
        return report

    def _say(self, message: str) -> None:
        """Print a message when verbose mode is enabled."""
        if self.verbose:
            print(message)

    @staticmethod
    def _scale_row_params(params: dict[str, Any], fidelity: float) -> dict[str, Any]:
        """Rescale absolute row-count parameters to the fidelity actually trained on.

        min_child_samples and its siblings count rows, so the same value is a far
        stronger constraint on a subsample than on the full data. Multiplying by the
        fidelity keeps sampled values in full-data units, so they transfer unchanged
        from the explore to the confirm stage and into best_params_.

        Args:
            params (dict[str, Any]): Parameters in full-data units.
            fidelity (float): Training rows at this fidelity divided by training rows
                at full fidelity. 1.0 leaves the parameters untouched.

        Returns:
            dict[str, Any]: Parameters adjusted for this fidelity.
        """
        if fidelity >= 1.0:
            return params
        scaled = dict(params)
        for name in ROW_SCALED_PARAM_NAMES:
            value = scaled.get(name)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                continue
            if isinstance(value, int):
                scaled[name] = max(1, int(round(value * fidelity)))
            else:
                scaled[name] = max(1e-6, float(value * fidelity))
        return scaled

    def _select_best(
        self, plan: _EvaluationPlan, estimator: BaseEstimator, study: optuna.study.Study
    ) -> None:
        """Choose between the searched configurations and the estimator's own parameters.

        The baseline guard runs first: only trials beating the baseline by more than one
        standard error of their paired per-fold differences are eligible. The one-standard-error tie
        rule then picks among the eligible trials only, so preferring a cheaper tied
        trial can never hand the win back to the baseline.

        Args:
            plan (_EvaluationPlan): Full-fidelity plan the baseline is scored on.
            estimator (BaseEstimator): Estimator template supplying the baseline.
            study (optuna.study.Study): Completed final-stage study.
        """
        baseline_params = self._baseline_params(estimator, snap=False)
        baseline_folds, baseline_iterations = self._evaluate_config(
            plan, estimator, baseline_params
        )
        self.baseline_score_ = self._penalized_score(baseline_folds)
        eligible = [
            trial
            for trial in self._completed_trials(study)
            if self._beats(
                trial.value,
                self.baseline_score_,
                self._required_margin(self._fold_scores(trial), baseline_folds),
            )
        ]
        if not eligible:
            winner = _Candidate(
                baseline_params,
                self.baseline_score_,
                baseline_iterations,
                BASELINE_LABEL,
            )
        else:
            chosen = self._select_confirm_trial(eligible)
            winner = _Candidate(
                dict(chosen.user_attrs.get("params", chosen.params)),
                float(chosen.value),
                list(chosen.user_attrs.get("best_iterations", [])),
                TUNED_LABEL,
            )
        self._commit_selection(estimator, plan, winner)

    def _select_confirm_trial(
        self, trials: list[optuna.trial.FrozenTrial]
    ) -> optuna.trial.FrozenTrial:
        """Apply the one-standard-error rule to a set of completed trials.

        When the top configurations are statistically tied, the argmax selects on noise
        and does not replicate. Among trials the best one does not beat by more than one
        standard error of their paired per-fold differences, the one with the fewest
        boosting rounds is returned.

        Args:
            trials (list[optuna.trial.FrozenTrial]): Completed trials to choose from.

        Returns:
            optuna.trial.FrozenTrial: The trial whose parameters should be returned.
        """
        pick = max if self.metric_direction == "maximize" else min
        best = pick(trials, key=lambda trial: trial.value)
        best_folds = self._fold_scores(best)
        ties = [
            trial
            for trial in trials
            if not self._beats(
                best.value,
                trial.value,
                self._required_margin(best_folds, self._fold_scores(trial)),
            )
        ]
        self.confirm_ties_ = len(ties)
        return min(
            ties, key=lambda trial: (self._median_iterations(trial), trial.number)
        )

    def _should_explore(self, n_rows: int) -> bool:
        """Decide whether a dataset is large enough to justify a cheap explore stage."""
        return (
            self.explore_fraction < 1.0
            and n_rows * self.explore_fraction > self.explore_min_rows
        )

    def _snap_candidate(
        self, candidate: dict[str, Any], space: ParamsGrid
    ) -> dict[str, Any]:
        """Fit a configuration into the space it is enqueued against.

        Optuna silently ignores an enqueued value its distribution excludes, so a warm
        start outside the narrowed box would quietly become a random trial.

        Args:
            candidate (dict[str, Any]): Configuration to enqueue.
            space (ParamsGrid): Space being searched.

        Returns:
            dict[str, Any]: Configuration guaranteed to lie inside the space.
        """
        snapped: dict[str, Any] = {}
        for name, value in candidate.items():
            if name not in space:
                continue
            param_type, values, options = self._unpack_spec(space[name])
            fixed = self._snap_to_space(param_type, values, options, value)
            if fixed is not None:
                snapped[name] = fixed
        return snapped

    @staticmethod
    def _snap_to_space(
        param_type: str,
        values: list[object],
        options: dict[str, Any],
        value: Any,
    ) -> Any:
        """Clamp and snap a value onto one grid entry.

        Library defaults routinely sit outside a hand-written grid (LightGBM's reg_alpha
        of 0.0 against a log range starting at 1e-3), and Optuna cannot enqueue a value
        its distribution does not admit.

        Args:
            param_type (str): One of "int", "float", or "categorical".
            values (list[object]): Bounds or choices from the grid.
            options (dict[str, Any]): Extra options such as step or log.
            value (Any): Value to snap.

        Returns:
            Any: A value inside the space, or None when it cannot be represented.
        """
        if param_type == "categorical":
            return value if value in values else None
        if param_type not in NUMERIC_PARAM_TYPES:
            return None
        low, high = values
        step = None if options.get("log", False) else options.get("step")
        snapped = min(max(float(value), low), high)
        if step:
            snapped = min(max(low + round((snapped - low) / step) * step, low), high)
        return int(round(snapped)) if param_type == "int" else float(snapped)

    def _split_early_stopping(
        self, y: pd.Series, train_idx: np.ndarray, fold_idx: int
    ) -> tuple[np.ndarray, np.ndarray | None]:
        """Carve a dedicated early-stopping split out of one training fold.

        Args:
            y (pd.Series): Target for the fidelity level being evaluated.
            train_idx (np.ndarray): Row positions of the training fold.
            fold_idx (int): Zero-based fold index, mixed into the split seed.

        Returns:
            tuple[np.ndarray, np.ndarray | None]: Fit positions and early-stopping
                positions, the latter None when the carve-out is disabled or the fold
                is too small to give rows up.
        """
        if self._n_early_stopping_rows(len(train_idx)) == 0:
            return train_idx, None
        return self._split_rows(
            train_idx, self.es_holdout_size, self.seed + fold_idx, y
        )

    def _split_rows(
        self, indices: np.ndarray, test_size: float | int, seed: int, y: pd.Series
    ) -> tuple[np.ndarray, np.ndarray]:
        """Split row positions in two, stratified on y for classification when possible.

        Args:
            indices (np.ndarray): Row positions to split.
            test_size (float | int): Fraction or row count of the second part.
            seed (int): Random seed of the split.
            y (pd.Series): Target, indexed by position.

        Returns:
            tuple[np.ndarray, np.ndarray]: The two parts of indices.
        """
        stratify = y.iloc[indices] if self.problem_type == "classification" else None
        try:
            first, second = train_test_split(
                indices, test_size=test_size, random_state=seed, stratify=stratify
            )
        except ValueError:
            first, second = train_test_split(
                indices, test_size=test_size, random_state=seed
            )
        return first, second

    def _subsample(
        self, X: pd.DataFrame, y: pd.Series
    ) -> tuple[pd.DataFrame, pd.Series]:
        """Draw the fixed explore-stage subsample, stratified for classification.

        Every explore trial sees the same rows, so comparisons between trials are paired
        rather than confounded by a fresh draw, which matters because TPE has no noise
        term. The winner's curse this invites is neutralised by the confirm stage.

        Args:
            X (pd.DataFrame): Full training features.
            y (pd.Series): Full training target.

        Returns:
            tuple[pd.DataFrame, pd.Series]: Subsampled features and target.
        """
        n_sample = min(
            len(X), max(self.explore_min_rows, int(len(X) * self.explore_fraction))
        )
        if n_sample >= len(X):
            return X, y
        _, sample_idx = self._split_rows(np.arange(len(X)), n_sample, self.seed, y)
        sample_idx = np.sort(sample_idx)
        return X.iloc[sample_idx].reset_index(drop=True), y.iloc[
            sample_idx
        ].reset_index(drop=True)

    def _top_params(self, study: optuna.study.Study) -> list[dict[str, Any]]:
        """Return the sampled params of the top_k completed trials of a study, best first."""
        ranked = sorted(
            self._completed_trials(study),
            key=lambda trial: trial.value,
            reverse=self.metric_direction == "maximize",
        )
        return [trial.params for trial in ranked[: self.top_k]]

    def _tunable_names(self) -> list[str]:
        """List the grid entries Optuna actually samples."""
        return [
            name for name, spec in self.params_grid.items() if spec[0] != "constant"
        ]

    @staticmethod
    def _unpack_spec(
        param_values: GridEntry,
    ) -> tuple[str, list[object], dict[str, Any]]:
        """Split a grid entry into its type, values, and keyword options."""
        options = param_values[2] if len(param_values) == 3 else {}
        return param_values[0], param_values[1], options

    def _validate_settings(self) -> None:
        """Reject staging options that cannot run.

        Raises:
            ValueError: If any staging option lies outside its valid range.
        """
        if self.n_parallel_trials < 1:
            raise ValueError("n_parallel_trials must be a positive integer.")
        if self.top_k < 1:
            raise ValueError("top_k must be a positive integer.")
        if not 0.0 < self.explore_fraction <= 1.0:
            raise ValueError("explore_fraction must lie in (0, 1].")
        if not 0.0 <= self.es_holdout_size < 1.0:
            raise ValueError("es_holdout_size must lie in [0, 1).")

    def _warn_reconciled(self, name: str, missing: list[str], kept: list[str]) -> None:
        """Report a realigned categorical declaration, once per parameter per search."""
        if not self.verbose or name in self._warned_categoricals:
            return
        self._warned_categoricals.add(name)
        print(
            f"⚠️  '{name}' named {len(missing)} column(s) the preprocessing pipeline does not "
            f"produce ({missing[:5]}); using {kept if kept else 'no'} categorical column(s) instead."
        )


if __name__ == "__main__":
    from lightgbm import LGBMClassifier
    from sklearn.datasets import make_classification

    from kvbiii_ml.data_processing.preprocessing.outlier_handling.winsorizer_trimmer import (
        WinsorizerWithOriginal,
    )

    RANDOM_STATE = 17
    N_SAMPLES = 2_000
    N_FEATURES = 10
    FEATURE_NAMES = [f"feature_{i}" for i in range(N_FEATURES)]

    def _run_demo() -> None:
        """Run RandomSearchCV on a synthetic classification dataset."""
        x_arr, y_arr = make_classification(
            n_samples=N_SAMPLES,
            n_features=N_FEATURES,
            n_informative=5,
            n_redundant=2,
            random_state=RANDOM_STATE,
        )
        x_df = pd.DataFrame(x_arr, columns=FEATURE_NAMES)
        y_ser = pd.Series(y_arr)
        preprocessing_pipeline = Pipeline(
            [
                (
                    "winsorizer",
                    WinsorizerWithOriginal(
                        variables=FEATURE_NAMES, capping_method="gaussian", tail="right"
                    ),
                ),
            ]
        )
        cross_validation_trainer = CrossValidationTrainer(
            metric_name="Balanced Accuracy",
            problem_type="classification",
            cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE),
            preprocessing_pipeline=preprocessing_pipeline,
            verbose=False,
        )
        tuner = RandomSearchCV(
            cross_validator=cross_validation_trainer,
            n_trials=10,
            seed=RANDOM_STATE,
            n_parallel_trials=2,
        )
        clf = LGBMClassifier(
            n_estimators=400,
            early_stopping_round=50,
            verbose=-1,
            n_jobs=1,
            random_state=RANDOM_STATE,
        )
        params_grid = {
            "verbose": ("constant", [-1]),
            "num_leaves": ("int", [16, 128]),
            "learning_rate": ("float", [0.01, 0.3], {"log": True}),
            "min_child_samples": ("int", [5, 50]),
        }
        tuner.tune(estimator=clf, X=x_df, y=y_ser, params_grid=params_grid)
        print(f"\nSelection source: {tuner.selection_source_}")
        print(f"Refit-ready params: {tuner.best_params_}")

    _run_demo()
