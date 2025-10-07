import numpy as np
import optuna
import warnings
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.model_selection import KFold

from kvbiii_ml.modeling.training.cross_validation import CrossValidationTrainer

warnings.filterwarnings(
    "ignore",
    message="Argument ``multivariate`` is an experimental feature. The interface can change in the future.",
    category=UserWarning,
    module="optuna",
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
            cross_validator (CrossValidationTrainer): Cross-validation splitter. Defaults to 5-fold KFold.
            n_trials (int): Number of trials to run. Defaults to 100.
            seed (int): Random seed for the sampler. Defaults to 17.
            allow_negative_weights (bool): Whether to allow negative blending weights. Defaults to False.
        """
        self.estimators = list(estimators)
        self.cross_validator = cross_validator
        self.n_trials = n_trials
        self.seed = seed
        self.allow_negative_weights = allow_negative_weights
        self.problem_type = cross_validator.problem_type
        self.metric_fn = self.cross_validator.metric_fn
        self.metric_type = self.cross_validator.metric_type
        self.metric_direction = self.cross_validator.metric_direction

        self.best_weights = None

    def tune(self, X: pd.DataFrame, y: pd.Series) -> optuna.study.Study:
        """Run Optuna to find the best blending weights.

        Args:
            X (pd.DataFrame): Feature matrix.
            y (pd.Series): Target vector.

        Returns:
            optuna.study.Study: Completed study. Best weights are stored in ``self.best_weights``.
        """
        X, y = self.check_X(X), self.check_y(y)
        y_true, preds_list = self._perform_cv(X, y)
        study = self._create_study()
        study.optimize(
            lambda trial: self._objective(trial, y_true, preds_list),
            n_trials=self.n_trials,
        )
        weights = np.array(
            [study.best_params[f"w{i}"] for i in range(len(self.estimators))],
            dtype=float,
        )
        if self.allow_negative_weights:
            l1 = np.sum(np.abs(weights))
            if l1 <= 1e-12:
                weights = np.zeros_like(weights, dtype=float)
                weights[0] = 1.0
            else:
                weights = weights / l1
        else:
            weights = weights / weights.sum()
        self.best_weights = weights
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
            study_name="Ensemble weight tuning",
        )

    def _objective(
        self,
        trial: optuna.Trial,
        y_true: pd.Series,
        preds_list: list[pd.Series | pd.DataFrame],
    ) -> float:
        """Objective function for weight tuning using cached predictions.

        Args:
            trial (optuna.Trial): Current Optuna trial.
            y_true (pd.Series): Validation targets with preserved indices.
            preds_list (list[pd.Series | pd.DataFrame]): Per-estimator predictions with preserved indices.

        Returns:
            float: Metric value computed on the blended predictions.
        """
        n_estimators = len(self.estimators)
        if self.allow_negative_weights:
            weights = np.array(
                [trial.suggest_float(f"w{i}", -1.0, 1.0) for i in range(n_estimators)],
                dtype=float,
            )
            l1 = np.sum(np.abs(weights))
            if l1 <= 1e-12:
                weights[0] = 1.0
                l1 = 1.0
            weights = weights / l1
        else:
            weights = np.array(
                [trial.suggest_float(f"w{i}", 0.01, 0.99) for i in range(n_estimators)],
                dtype=float,
            )
            weights /= weights.sum()

        blended = self._blend_predictions(preds_list, weights)

        if self.problem_type == "classification" and self.metric_type == "preds":
            if isinstance(blended, pd.Series):
                blended = (blended >= 0.5).astype(int)
            else:
                blended = blended.idxmax(axis=1)

        return float(self.metric_fn(y_true, blended))

    def _blend_predictions(
        self, preds_list: list[pd.Series | pd.DataFrame], weights: np.ndarray
    ) -> pd.Series | pd.DataFrame:
        """Blend predictions with weights while preserving original indices.

        Args:
            preds_list (list[pd.Series | pd.DataFrame]): Per-estimator predictions.
            weights (np.ndarray): Weights for blending.

        Returns:
            pd.Series | pd.DataFrame: Blended predictions with original indices.
        """
        first_pred = preds_list[0]
        is_clf_probs = (
            self.problem_type == "classification" and self.metric_type == "probs"
        )
        if isinstance(first_pred, pd.Series):
            if is_clf_probs and self.allow_negative_weights:
                eps = 1e-9
                logits_list = [
                    np.log(
                        np.clip(pred.values, eps, 1 - eps)
                        / np.clip(1 - pred.values, eps, 1 - eps)
                    )
                    for pred in preds_list
                ]
                blended_logits = np.zeros_like(logits_list[0])
                for w, logit in zip(weights, logits_list):
                    blended_logits += w * logit
                blended_values = 1.0 / (1.0 + np.exp(-blended_logits))
            else:
                blended_values = np.zeros_like(first_pred.values)
                for w, pred in zip(weights, preds_list):
                    blended_values += w * pred.values
                if is_clf_probs:
                    blended_values = np.clip(blended_values, 0.0, 1.0)
            return pd.Series(blended_values, index=first_pred.index)
        else:
            if is_clf_probs and self.allow_negative_weights:
                eps = 1e-12
                logp_list = [
                    np.log(np.clip(pred.values, eps, 1.0)) for pred in preds_list
                ]
                scores = np.zeros_like(logp_list[0])
                for w, logp in zip(weights, logp_list):
                    scores += w * logp
                scores -= scores.max(axis=1, keepdims=True)
                exp_scores = np.exp(scores)
                blended_values = exp_scores / np.clip(
                    exp_scores.sum(axis=1, keepdims=True), eps, None
                )
            else:
                blended_values = np.zeros_like(first_pred.values)
                for w, pred in zip(weights, preds_list):
                    blended_values += w * pred.values
                row_sums = blended_values.sum(axis=1, keepdims=True)
                blended_values = blended_values / np.where(
                    row_sums == 0.0, 1.0, row_sums
                )
                blended_values = np.clip(blended_values, 0.0, 1.0)
                blended_values /= np.clip(
                    blended_values.sum(axis=1, keepdims=True), 1e-12, None
                )
            return pd.DataFrame(
                blended_values, index=first_pred.index, columns=first_pred.columns
            )

    def _perform_cv(
        self, X: pd.DataFrame, y: pd.Series
    ) -> tuple[pd.Series, list[pd.Series | pd.DataFrame]]:
        """Collect validation predictions for each estimator across CV folds.

        Args:
            X (pd.DataFrame): Feature matrix.
            y (pd.Series): Target vector.

        Returns:
            tuple[pd.Series, list[pd.Series | pd.DataFrame]]: Tuple with true validation targets and per-estimator predictions.
        """
        y_valid_true = pd.Series([], dtype=y.dtype)
        preds_per_estimator = [[] for _ in range(len(self.estimators))]

        for estimator_idx, estimator in enumerate(self.estimators):
            self.cross_validator.fit(estimator, X, y)

            for fold_idx, (_, valid_idx) in enumerate(
                self.cross_validator.cv.split(X, y)
            ):
                X_valid = X.iloc[valid_idx]
                y_valid = y.iloc[valid_idx]
                if estimator_idx == 0:
                    y_valid_true = pd.concat([y_valid_true, y_valid])
                fitted_estimator = self.cross_validator.fitted_estimators_[fold_idx]
                if self.problem_type == "classification":
                    y_valid_pred = fitted_estimator.predict_proba(X_valid)
                    if y_valid_pred.shape[1] == 2:
                        y_pred_fold = pd.Series(y_valid_pred[:, 1], index=X_valid.index)
                    else:
                        y_pred_fold = pd.DataFrame(y_valid_pred, index=X_valid.index)
                else:
                    y_valid_pred = fitted_estimator.predict(X_valid)
                    y_pred_fold = pd.Series(y_valid_pred, index=X_valid.index)

                preds_per_estimator[estimator_idx].append(y_pred_fold)
        combined_preds = []
        for estimator_preds in preds_per_estimator:
            first_pred = estimator_preds[0]
            if isinstance(first_pred, pd.Series):
                combined_pred = pd.concat(estimator_preds)
            else:  # DataFrame
                combined_pred = pd.concat(estimator_preds)
            combined_preds.append(combined_pred)

        return y_valid_true, combined_preds

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
    import pandas as pd
    import numpy as np
    from sklearn.datasets import make_classification
    from sklearn.linear_model import LogisticRegression

    X_arr, y_arr = make_classification(
        n_samples=3000, n_features=10, n_informative=5, n_redundant=2, random_state=17
    )
    X_df = pd.DataFrame(X_arr)
    y_ser = pd.Series(y_arr)

    estimators = [
        LogisticRegression(max_iter=1000, solver="liblinear", random_state=17),
        LogisticRegression(C=0.5, max_iter=1000, solver="liblinear", random_state=17),
        LogisticRegression(C=2.0, max_iter=1000, solver="liblinear", random_state=17),
        LogisticRegression(C=0.1, max_iter=1000, solver="liblinear", random_state=17),
        LogisticRegression(C=5.0, max_iter=1000, solver="liblinear", random_state=17),
        LogisticRegression(C=0.01, max_iter=1000, solver="liblinear", random_state=17),
        LogisticRegression(C=10.0, max_iter=1000, solver="liblinear", random_state=17),
        LogisticRegression(C=0.001, max_iter=1000, solver="liblinear", random_state=17),
    ]

    cross_validator = CrossValidationTrainer(
        metric_name="Roc AUC",
        problem_type="classification",
        cv=KFold(n_splits=5, shuffle=True, random_state=17),
        processors=None,
        verbose=False,
    )

    tuner = EnsembleWeightTunerCV(
        estimators=estimators,
        cross_validator=cross_validator,
        n_trials=100,
        seed=17,
        allow_negative_weights=True,
    )
    study = tuner.tune(X_df, y_ser)
    print("Best trial value:", study.best_value)
    print("Best weights:", tuner.best_weights)

    from sklearn.datasets import make_regression
    from sklearn.linear_model import LinearRegression

    X_reg, y_reg = make_regression(
        n_samples=300, n_features=10, n_informative=5, noise=0.1, random_state=17
    )
    X_reg_df = pd.DataFrame(X_reg)
    y_reg_ser = pd.Series(y_reg)
    reg_estimators = [
        LinearRegression(),
        LinearRegression(fit_intercept=False),
    ]
    reg_cross_validator = CrossValidationTrainer(
        metric_name="MSE",
        problem_type="regression",
        cv=KFold(n_splits=5, shuffle=True, random_state=17),
        processors=None,
        verbose=False,
    )
    reg_tuner = EnsembleWeightTunerCV(
        estimators=reg_estimators,
        cross_validator=reg_cross_validator,
        n_trials=50,
        seed=17,
        allow_negative_weights=False,
    )
    reg_study = reg_tuner.tune(X_reg_df, y_reg_ser)
    print("Best trial value for regression:", reg_study.best_value)
    print("Best weights for regression:", reg_tuner.best_weights)
