import warnings

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
    """

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
        self.problem_type = cross_validator.problem_type
        self.metric_fn = cross_validator.metric_fn
        self.metric_type = cross_validator.metric_type
        self.metric_direction = cross_validator.metric_direction
        self.best_weights: np.ndarray | None = None

    def tune(self, X: pd.DataFrame, y: pd.Series) -> optuna.study.Study:
        """Run Optuna to find the best blending weights.

        Args:
            X (pd.DataFrame): Feature matrix.
            y (pd.Series): Target vector.

        Returns:
            optuna.study.Study: Completed study. Best weights stored in ``self.best_weights``.
        """
        X, y = self.check_X(X), self.check_y(y)
        y_true, preds_list = self._perform_cv(X, y)
        study = self._create_study()
        study.optimize(
            lambda trial: self._objective(trial, y_true, preds_list),
            n_trials=self.n_trials,
        )
        n = len(self.estimators)
        raw_weights = np.array(
            [study.best_params[f"w{i}"] for i in range(n)], dtype=float
        )
        self.best_weights = self._normalize_weights(raw_weights)
        return study

    def _normalize_weights(self, weights: np.ndarray) -> np.ndarray:
        """Normalize weights to sum to 1 (or L1-norm to 1 for negative weights).

        Args:
            weights (np.ndarray): Raw weight vector.

        Returns:
            np.ndarray: Normalized weight vector. Falls back to [1, 0, ...] if norm is near zero.
        """
        if self.allow_negative_weights:
            l1 = np.abs(weights).sum()
            if l1 <= 1e-12:
                result = np.zeros_like(weights)
                result[0] = 1.0
                return result
            return weights / l1
        return weights / weights.sum()

    def _create_study(self) -> optuna.study.Study:
        """Create an Optuna study with TPE sampler and Hyperband pruner.

        Returns:
            optuna.study.Study: Configured study.
        """
        sampler = optuna.samplers.TPESampler(
            seed=self.seed, n_startup_trials=25, multivariate=True
        )
        return optuna.create_study(
            direction=self.metric_direction,
            sampler=sampler,
            pruner=optuna.pruners.HyperbandPruner(),
            study_name="Ensemble weight tuning",
        )

    def _objective(
        self,
        trial: optuna.Trial,
        y_true: pd.Series,
        preds_list: list[pd.Series | pd.DataFrame],
    ) -> float:
        """Objective function for weight tuning using cached OOF predictions.

        Args:
            trial (optuna.Trial): Current Optuna trial.
            y_true (pd.Series): Validation targets with preserved indices.
            preds_list (list[pd.Series | pd.DataFrame]): Per-estimator OOF predictions.

        Returns:
            float: Metric value computed on the blended predictions.
        """
        n = len(self.estimators)
        lo, hi = (-1.0, 1.0) if self.allow_negative_weights else (0.01, 0.99)
        weights = self._normalize_weights(
            np.array(
                [trial.suggest_float(f"w{i}", lo, hi) for i in range(n)], dtype=float
            )
        )
        blended = self._blend_predictions(preds_list, weights)
        if self.problem_type == "classification" and self.metric_type == "preds":
            blended = (
                blended.idxmax(axis=1)
                if isinstance(blended, pd.DataFrame)
                else (blended >= 0.5).astype(int)
            )
        return float(self.metric_fn(y_true, blended))

    def _blend_predictions(
        self, preds_list: list[pd.Series | pd.DataFrame], weights: np.ndarray
    ) -> pd.Series | pd.DataFrame:
        """Blend predictions with weights while preserving original indices.

        For negative weights with classification probabilities, uses logit (binary)
        or log-softmax (multiclass) blending to keep outputs in valid probability space.
        Positive-weight blending uses a direct weighted sum with optional row renormalization
        for multiclass outputs.

        Args:
            preds_list (list[pd.Series | pd.DataFrame]): Per-estimator OOF predictions.
            weights (np.ndarray): Normalized blending weights.

        Returns:
            pd.Series | pd.DataFrame: Blended predictions with original indices.
        """
        first = preds_list[0]
        is_df = isinstance(first, pd.DataFrame)
        neg_clf = self.problem_type == "classification" and self.allow_negative_weights
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

    def _perform_cv(
        self, X: pd.DataFrame, y: pd.Series
    ) -> tuple[pd.Series, list[pd.Series | pd.DataFrame]]:
        """Collect OOF predictions for each estimator across CV folds.

        Applies each fold's fitted pipeline to the validation split before calling
        the estimator, ensuring consistency with training-time transformations.
        If no pipeline is defined on the cross_validator, validation data is passed
        through unchanged.

        Args:
            X (pd.DataFrame): Feature matrix.
            y (pd.Series): Target vector.

        Returns:
            tuple[pd.Series, list[pd.Series | pd.DataFrame]]: OOF true targets and
                per-estimator concatenated OOF predictions.
        """
        splits = list(self.cross_validator.cv.split(X, y))
        y_valid_true = pd.concat([y.iloc[val_idx] for _, val_idx in splits])
        preds_per_estimator = []

        for estimator in self.estimators:
            self.cross_validator.fit(estimator, X, y)
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

            preds_per_estimator.append(pd.concat(est_preds))

        return y_valid_true, preds_per_estimator

    @staticmethod
    def check_X(X: pd.DataFrame | np.ndarray | list | dict) -> pd.DataFrame:
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
    clf_study = tuner_clf.tune(X_df, y_ser)
    print("Best trial value:", clf_study.best_value)
    print("Best weights:", tuner_clf.best_weights)

    print("\n=== Regression ===")
    X_reg, y_reg = make_regression(
        n_samples=N_SAMPLES,
        n_features=N_FEATURES,
        n_informative=5,
        noise=0.1,
        random_state=RANDOM_STATE,
    )
    X_reg_df = pd.DataFrame(X_reg, columns=FEATURE_NAMES)
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
    reg_study = tuner_reg.tune(X_reg_df, y_reg_ser)
    print("Best trial value:", reg_study.best_value)
    print("Best weights:", tuner_reg.best_weights)
