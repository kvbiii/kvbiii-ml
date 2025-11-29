import warnings
import numpy as np
import pandas as pd
import optuna
from sklearn.base import BaseEstimator
from sklearn.model_selection import KFold
from kvbiii_ml.modeling.training.cross_validation import CrossValidationTrainer

warnings.filterwarnings(
    "ignore", category=optuna.exceptions.ExperimentalWarning, module="optuna"
)


class CutoffTunerCV:
    """Tune decision cutoffs for classification using cross-validation.

    This tuner fits the estimator across CV folds, caches validation
    probabilities, and uses Optuna to optimize per-class cutoffs that
    convert probabilities into final class predictions.
    """

    def __init__(
        self,
        estimator: BaseEstimator,
        cross_validator: CrossValidationTrainer,
        n_trials: int = 50,
        seed: int = 17,
    ) -> None:
        """Initialize the cutoff tuner.

        Args:
            estimator (BaseEstimator): Classifier whose probabilities will be thresholded.
            cross_validator (CrossValidationTrainer): Cross-validation splitter. Defaults to 5-fold KFold.
            n_trials (int): Number of trials to run. Defaults to 100.
            seed (int): Random seed for the sampler. Defaults to 17.
        """
        self.estimator = estimator
        self.cross_validator = cross_validator
        self.n_trials = n_trials
        self.seed = seed
        self.metric_fn = self.cross_validator.metric_fn
        self.metric_direction = self.cross_validator.metric_direction
        self.fitted_estimators_: list = []
        self.best_cutoffs: np.ndarray | None = None

    def tune(self, X: pd.DataFrame, y: pd.Series) -> optuna.study.Study:
        """Run Optuna to find cutoffs that maximize the chosen metric.

        Args:
            X (pd.DataFrame): Feature matrix.
            y (pd.Series): Target vector.

        Returns:
            optuna.study.Study: Completed study with best cutoffs stored in ``self.best_cutoffs``.
        """
        X, y = self.check_X(X), self.check_y(y)
        y_true, y_proba = self._perform_cv(X, y)
        study = self._create_study()
        study.optimize(
            lambda trial: self._objective(trial, y_true, y_proba),
            n_trials=self.n_trials,
        )
        if y_proba.ndim == 1:
            self.best_cutoffs = np.array([study.best_params["cutoff"]], dtype=float)
        else:
            n_classes = y_proba.shape[1]
            self.best_cutoffs = np.array(
                [study.best_params[f"cutoff_{i}"] for i in range(n_classes)],
                dtype=float,
            )
            self.best_cutoffs = self.best_cutoffs / self.best_cutoffs.sum()
        return study

    def _create_study(self) -> optuna.study.Study:
        """Create an Optuna study with TPE sampler and Hyperband pruner.

        Returns:
            optuna.study.Study: Study configured with direction, sampler, and pruner.
        """
        sampler = optuna.samplers.TPESampler(
            seed=self.seed, n_startup_trials=25, multivariate=True
        )
        pruner = optuna.pruners.HyperbandPruner()
        return optuna.create_study(
            direction=self.metric_direction,
            sampler=sampler,
            pruner=pruner,
            study_name="Cutoff tuning",
        )

    def _objective(
        self, trial: optuna.Trial, y_true: np.ndarray, y_proba: np.ndarray
    ) -> float:
        """Objective to evaluate a set of cutoffs on cached probabilities.

        Args:
            trial (optuna.Trial): Current Optuna trial.
            y_true (np.ndarray): Concatenated validation targets across folds.
            y_proba (np.ndarray): Concatenated validation probabilities across folds.

        Returns:
            float: Metric value computed on predictions derived from the cutoffs.
        """
        eps = 1e-9
        if y_proba.ndim == 1:
            cutoff = trial.suggest_float("cutoff", 0.01, 0.99)
            y_pred = (y_proba >= cutoff).astype(int)
        else:
            n_classes = y_proba.shape[1]
            cutoffs = np.array(
                [
                    trial.suggest_float(f"cutoff_{i}", 0.01, 0.99)
                    for i in range(n_classes)
                ],
                dtype=float,
            )
            cutoffs = cutoffs / np.clip(cutoffs.sum(), eps, None)
            scores = y_proba / np.clip(cutoffs.reshape(1, -1), eps, None)
            y_pred = np.argmax(scores, axis=1)
        return float(self.metric_fn(y_true, y_pred))

    def _perform_cv(
        self, X: pd.DataFrame, y: pd.Series
    ) -> tuple[np.ndarray, np.ndarray]:
        """Perform cross-validation to cache probabilities for cutoff tuning.

        Args:
            X (pd.DataFrame): Feature matrix.
            y (pd.Series): Target vector.

        Returns:
            tuple[np.ndarray, np.ndarray]: Tuple of true labels and predicted probabilities.
        """
        self.cross_validator.fit(self.estimator, X, y)
        self.fitted_estimators_ = self.cross_validator.fitted_estimators_
        y_valid_true: list[float] = []
        y_valid_proba_list: list[np.ndarray] = []
        for fold_idx, (_, valid_idx) in enumerate(self.cross_validator.cv.split(X, y)):
            X_valid = X.iloc[valid_idx]
            y_valid = y.iloc[valid_idx]
            fitted_estimator = self.fitted_estimators_[fold_idx]
            y_valid_proba = fitted_estimator.predict_proba(X_valid)
            y_valid_true.append(y_valid)
            y_valid_proba_list.append(y_valid_proba)
        y_valid_true = np.concatenate(y_valid_true)
        if all(proba.ndim == 2 and proba.shape[1] == 2 for proba in y_valid_proba_list):
            y_valid_proba_list = [proba[:, 1] for proba in y_valid_proba_list]
        y_valid_proba = np.concatenate(y_valid_proba_list, axis=0)
        return y_valid_true, y_valid_proba

    @staticmethod
    def check_X(X: object) -> pd.DataFrame:
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
    from sklearn.datasets import make_classification
    from sklearn.linear_model import LogisticRegression

    X_arr, y_arr = make_classification(
        n_samples=300, n_features=10, n_informative=5, n_redundant=2, random_state=17
    )
    X_df = pd.DataFrame(X_arr)
    y_ser = pd.Series(y_arr)

    cross_validation_trainer = CrossValidationTrainer(
        metric_name="Accuracy",
        problem_type="classification",
        cv=KFold(n_splits=5, shuffle=True, random_state=17),
        processors=None,
        verbose=False,
    )

    clf = LogisticRegression(max_iter=1000, solver="liblinear", random_state=17)

    tuner = CutoffTunerCV(
        estimator=clf,
        cross_validator=cross_validation_trainer,
        n_trials=100,
    )
    cutoff_study = tuner.tune(X_df, y_ser)
    print("Best trial value:", cutoff_study.best_value)
    print("Best cutoffs:", tuner.best_cutoffs)
