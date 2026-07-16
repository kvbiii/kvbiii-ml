import warnings

import numpy as np
import optuna
import pandas as pd
from sklearn.base import BaseEstimator, clone

from kvbiii_ml.modeling.training.base_trainer import BaseTrainer
from kvbiii_ml.modeling.training.cross_validation import CrossValidationTrainer

warnings.filterwarnings(
    "ignore",
    category=optuna.exceptions.ExperimentalWarning,
    message=".*multivariate.*",
)


class RandomSearchCV:
    """Lightweight random search tuner using Optuna.

    This class performs randomized hyperparameter search using Optuna and
    reuses project utilities for metrics, training, and cross-validation.
    """

    def __init__(
        self,
        cross_validator: CrossValidationTrainer,
        n_trials: int = 100,
        seed: int = 17,
        max_samples: int = 100000,
    ):
        """Initialize a lightweight random search tuner.

        Args:
            cross_validator (BaseCrossValidator): Cross-validation splitter. Defaults to 5-fold KFold.
            n_trials (int): Number of trials to run. Defaults to 100.
            seed (int): Random seed for the sampler. Defaults to 17.
            max_samples (int): Maximum number of rows used during CV objective evaluation.
                Defaults to 100000.
        """
        self.cross_validator = cross_validator
        self.n_trials = n_trials
        self.seed = seed
        self.max_samples = max_samples
        self.metric_fn = self.cross_validator.metric_fn
        self.metric_type = self.cross_validator.metric_type
        self.metric_direction = self.cross_validator.metric_direction

    def tune(
        self,
        estimator: BaseEstimator,
        X: pd.DataFrame,
        y: pd.Series,
        params_grid: dict[str, tuple[str, list[object]]],
        X_valid: pd.DataFrame | None = None,
        y_valid: pd.Series | None = None,
    ) -> optuna.study.Study:
        """Run random search with Optuna on a hold-out or via cross-validation.

        Args:
            estimator (BaseEstimator): Estimator to tune.
            X (pd.DataFrame): Training features.
            y (pd.Series): Training target.
            params_grid (dict[str, tuple[str, list[object]]]): Parameter search space.
            X_valid (pd.DataFrame | None): Optional validation features.
            y_valid (pd.Series | None): Optional validation target.

        Returns:
            optuna.study.Study: The configured and optimized study.
        """
        self.params_grid = params_grid
        study = self.create_study()
        X, y = self.check_x(X), self.check_y(y)
        if X_valid is not None and y_valid is not None:
            X_valid = self.check_x(X_valid)
            y_valid = self.check_y(y_valid)
            study.optimize(
                lambda trial: self.objective(trial, estimator, X, y, X_valid, y_valid),
                n_trials=self.n_trials,
            )
        else:
            study.optimize(
                lambda trial: self.objective_cv(trial, estimator, X, y),
                n_trials=self.n_trials,
            )
        return study

    def create_study(self) -> optuna.study.Study:
        """Create an Optuna study with a seeded TPE sampler.

        Returns:
            optuna.study.Study: Configured study object.
        """
        sampler = optuna.samplers.TPESampler(
            seed=self.seed, n_startup_trials=25, multivariate=True
        )
        pruner = optuna.pruners.HyperbandPruner()
        return optuna.create_study(
            direction=self.metric_direction,
            sampler=sampler,
            pruner=pruner,
            study_name="Optuna tuning",
        )

    def get_param(
        self,
        trial: optuna.Trial,
        param_name: str,
        param_values: tuple[str, list[object]],
    ) -> object:
        """Sample a single hyperparameter from the provided search space.

        Args:
            trial (optuna.Trial): Active Optuna trial.
            param_name (str): Name of the hyperparameter.
            param_values (tuple[str, list[object]]): Tuple of (type, values).
                Supported types: "int", "float", "categorical", "constant".

        Returns:
            object: Sampled value compatible with the estimator.
        """
        if len(param_values) == 3:
            param_type, param_value, param_kwargs = param_values
        else:
            param_type, param_value = param_values
            param_kwargs = {}

        if param_type == "int":
            low, high = param_value
            step = param_kwargs.get("step", 1)
            return trial.suggest_int(param_name, low, high, step=step)
        elif param_type == "float":
            low, high = param_value
            log = param_kwargs.get("log", False)
            step = param_kwargs.get("step", None)
            if step is not None:
                return trial.suggest_float(param_name, low, high, log=log, step=step)
            else:
                return trial.suggest_float(param_name, low, high, log=log)
        elif param_type == "categorical":
            return trial.suggest_categorical(param_name, param_value)
        elif param_type == "constant":
            return param_value[0]
        else:
            raise ValueError(f"Unknown param_type: {param_type}")

    def objective(
        self,
        trial: optuna.Trial,
        estimator: BaseEstimator,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_valid: pd.DataFrame,
        y_valid: pd.Series,
    ) -> float:
        """Objective function for hold-out tuning.

        Clones the base estimator, sets parameters suggested by the trial,
        and evaluates on the provided validation split using project helpers.

        Args:
            trial (optuna.Trial): Active Optuna trial.
            X_train (pd.DataFrame): Training features.
            y_train (pd.Series): Training target.
            X_valid (pd.DataFrame): Validation features.
            y_valid (pd.Series): Validation target.

        Returns:
            float: Validation score for this trial.
        """
        params = {k: self.get_param(trial, k, v) for k, v in self.params_grid.items()}
        est = clone(estimator)
        est.set_params(**params)
        _, y_valid_pred, _, _ = BaseTrainer.fit_and_predict(
            est, X_train, y_train, X_valid, y_valid, self.metric_type
        )
        return float(self.metric_fn(y_valid, y_valid_pred))

    def objective_cv(
        self,
        trial: optuna.Trial,
        estimator: BaseEstimator,
        X: pd.DataFrame,
        y: pd.Series,
    ) -> float:
        """Objective function for cross-validated tuning.

        Samples parameters, clones the estimator, and evaluates mean validation
        score via CrossValidationTrainer on a size-limited sample for speed.

        Args:
            trial (optuna.Trial): Active Optuna trial.
            X (pd.DataFrame): Full training features.
            y (pd.Series): Full training target.

        Returns:
            float: Mean cross-validated validation score for this trial.
        """
        params = {k: self.get_param(trial, k, v) for k, v in self.params_grid.items()}
        if X.shape[0] > self.max_samples:
            x_sample = X.sample(n=self.max_samples, random_state=self.seed)
        else:
            x_sample = X.copy()
        y_sample = y.loc[x_sample.index]

        est = clone(estimator)
        est.set_params(**params)

        _, valid_scores, _ = self.cross_validator.fit(est, x_sample, y_sample)
        return float(np.mean(valid_scores))

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
                        variables=FEATURE_NAMES,
                        capping_method="gaussian",
                        tail="right",
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
            cross_validator=cross_validation_trainer, n_trials=10, seed=RANDOM_STATE
        )
        clf = LGBMClassifier(n_estimators=100, verbose=-1, random_state=RANDOM_STATE)
        params_grid = {
            "num_leaves": ("int", [16, 128]),
            "learning_rate": ("float", [0.01, 0.3], {"log": True}),
            "min_child_samples": ("int", [5, 50]),
        }
        study = tuner.tune(estimator=clf, X=x_df, y=y_ser, params_grid=params_grid)
        print("Best trial:")
        print(study.best_trial)
        print("Best params:")
        print(study.best_params)
        print("Best value:")
        print(study.best_value)

    _run_demo()
