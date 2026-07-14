import warnings

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
    """

    def __init__(
        self,
        estimator: BaseEstimator,
        cross_validator: CrossValidationTrainer,
        n_trials: int = 100,
        seed: int = 17,
    ) -> None:
        """Initialize the tuner.

        Args:
            estimator (BaseEstimator): Unfitted estimator to train during CV.
            cross_validator (CrossValidationTrainer): Configured CV trainer.
            n_trials (int): Number of Optuna trials. Defaults to 100.
            seed (int): Random seed for the TPE sampler. Defaults to 17.
        """
        self.estimator = estimator
        self.cross_validator = cross_validator
        self.n_trials = n_trials
        self.seed = seed
        self.metric_fn = cross_validator.metric_fn
        self.metric_direction = cross_validator.metric_direction
        self.fitted_estimators_: list[BaseEstimator] = []
        self.best_cutoffs: np.ndarray | None = None
        self._is_binary: bool = True

    def tune(self, X: pd.DataFrame, y: pd.Series) -> optuna.study.Study:
        """Fit CV, cache OOF probabilities, then optimise cutoffs.

        Args:
            X (pd.DataFrame): Feature matrix.
            y (pd.Series): Target vector.

        Returns:
            optuna.study.Study: Completed study; best cutoffs stored in ``self.best_cutoffs``.
        """
        X, y = self._check_X(X), self._check_y(y)
        y_true, y_proba = self._perform_cv(X, y)

        study = self._create_study()
        study.optimize(
            lambda trial: self._objective(trial, y_true, y_proba),
            n_trials=self.n_trials,
        )

        best = study.best_params
        if self._is_binary:
            self.best_cutoffs = np.array([best["cutoff"]], dtype=float)
        else:
            n_classes = y_proba.shape[1]
            self.best_cutoffs = np.array(
                [best[f"cutoff_{i}"] for i in range(n_classes)], dtype=float
            )
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
            X_proc = CrossValidationTrainer._transform_with_pipeline(fold_pipeline, X)
            y_proba = self.fitted_estimators_[0].predict_proba(X_proc)
        return self._apply_cutoffs(y_proba, self.best_cutoffs, self._is_binary)

    def _create_study(self) -> optuna.study.Study:
        """Create an Optuna study with multivariate TPE sampler.

        Returns:
            optuna.study.Study: Configured study.
        """
        sampler = optuna.samplers.TPESampler(
            seed=self.seed, n_startup_trials=25, multivariate=True
        )
        return optuna.create_study(
            direction=self.metric_direction,
            sampler=sampler,
            study_name="cutoff_tuning",
        )

    def _objective(
        self, trial: optuna.Trial, y_true: np.ndarray, y_proba: np.ndarray
    ) -> float:
        """Objective function for cutoff optimisation.

        Args:
            trial (optuna.Trial): Current Optuna trial.
            y_true (np.ndarray): OOF ground-truth labels.
            y_proba (np.ndarray): OOF probabilities.

        Returns:
            float: Metric value for the suggested cutoffs.
        """
        if self._is_binary:
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
            y_pred = self._apply_cutoffs(y_proba, cutoffs, is_binary=False)
        return float(self.metric_fn(y_true, y_pred))

    @staticmethod
    def _apply_cutoffs(
        y_proba: np.ndarray, cutoffs: np.ndarray, is_binary: bool
    ) -> np.ndarray:
        """Convert probabilities and cutoffs into class predictions.

        Args:
            y_proba (np.ndarray): (n_samples,) for binary or (n_samples, n_classes).
            cutoffs (np.ndarray): Per-class cutoff values.
            is_binary (bool): Whether to use scalar binary thresholding.

        Returns:
            np.ndarray: Integer class predictions of shape (n_samples,).
        """
        if is_binary:
            return (y_proba >= cutoffs[0]).astype(int)
        return np.argmax(y_proba / np.clip(cutoffs, 1e-9, None), axis=1)

    def _perform_cv(
        self, X: pd.DataFrame, y: pd.Series
    ) -> tuple[np.ndarray, np.ndarray]:
        """Run CV and return concatenated OOF labels and probabilities.

        Applies each fold's fitted pipeline to the validation split before
        calling predict_proba, matching training-time transformations.

        Args:
            X (pd.DataFrame): Feature matrix.
            y (pd.Series): Target vector.

        Returns:
            tuple[np.ndarray, np.ndarray]: OOF true labels and probabilities.
        """
        self.cross_validator.fit(self.estimator, X, y)
        self.fitted_estimators_ = self.cross_validator.fitted_estimators_

        oof_true, oof_proba = [], []
        for (_, val_idx), est, fold_pipeline in zip(
            self.cross_validator.cv.split(X, y),
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
        return y_true, y_proba_full[:, 1] if self._is_binary else y_proba_full

    @staticmethod
    def _check_X(X: object) -> pd.DataFrame:
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
    X_arr, y_arr = make_classification(
        n_samples=N_SAMPLES,
        n_features=N_FEATURES,
        n_informative=5,
        n_redundant=2,
        random_state=RANDOM_STATE,
    )
    X_df = pd.DataFrame(X_arr, columns=FEATURE_NAMES)
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
    study_bin = tuner_bin.tune(X_df, y_ser)
    print("Best trial value:", study_bin.best_value)
    print("Best cutoffs:", tuner_bin.best_cutoffs)

    print("\n=== Multiclass classification ===")
    X_arr_mc, y_arr_mc = make_classification(
        n_samples=N_SAMPLES,
        n_features=N_FEATURES,
        n_informative=5,
        n_redundant=2,
        n_classes=3,
        n_clusters_per_class=1,
        random_state=RANDOM_STATE,
    )
    X_df_mc = pd.DataFrame(X_arr_mc, columns=FEATURE_NAMES)
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
    study_mc = tuner_mc.tune(X_df_mc, y_ser_mc)
    print("Best trial value:", study_mc.best_value)
    print("Best cutoffs:", tuner_mc.best_cutoffs)
