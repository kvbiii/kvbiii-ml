import gc

import numpy as np
import pandas as pd
import shap
from sklearn.base import BaseEstimator


def compute_shap_values(
    model: object,
    X: pd.DataFrame,
    feature_names: list[str] | None = None,
) -> shap.Explanation:
    """Computes SHAP values for a model or ensemble with optimized performance.

    Args:
        model (object): Trained model or ensemble of models.
        X (pd.DataFrame): Input data for which to compute SHAP values.
        feature_names (list[str], optional): Names of the features in X.
            If None, uses X.columns. Defaults to None.

    Returns:
        shap.Explanation: SHAP explanation object with values for X.

    Raises:
        ValueError: If HistGradientBoosting models are detected (not supported).
    """
    feature_names = feature_names or list(X.columns)
    fitted_estimators = getattr(model, "fitted_estimators_", None)

    return (
        _compute_ensemble_shap(model, X, feature_names)
        if fitted_estimators
        else _compute_single_model_shap(model, X)
    )


def _compute_single_model_shap(model: object, X: pd.DataFrame) -> shap.Explanation:
    """Computes SHAP values for a single model.

    Args:
        model (object): Trained model to compute SHAP values for.
        X (pd.DataFrame): Input data for SHAP computation.

    Returns:
        shap.Explanation: SHAP explanation object for the model.

    Raises:
        ValueError: If model is HistGradientBoosting type (not supported).
    """
    if model.__class__.__name__.startswith("HistGradientBoosting"):
        raise ValueError(
            "HistGradientBoosting is not supported by SHAP. It requires conversion of "
            "categories to numerical codes, which results in loss of categorical variables "
            "interpretation, making SHAP values uninterpretable."
        )

    x_ordered = _order_x_for_estimator(X, model)
    explainer = shap.TreeExplainer(model, feature_perturbation="tree_path_dependent")

    try:
        return explainer(x_ordered)
    except AttributeError:
        return explainer(x_ordered.values)
    finally:
        gc.collect()


def _order_x_for_estimator(X: pd.DataFrame, estimator: BaseEstimator) -> pd.DataFrame:
    """Reorder columns to match the estimator's expected feature order.

    Args:
        X (pd.DataFrame): Input features.
        estimator (BaseEstimator): Trained estimator.

    Returns:
        pd.DataFrame: Reordered features if the estimator exposes feature
        names; otherwise returns X unchanged.
    """
    feature_names = getattr(estimator, "feature_names_in_", None) or getattr(
        estimator, "feature_names_", None
    )
    return X[feature_names] if feature_names is not None else X


def _compute_ensemble_shap(
    model: object, X: pd.DataFrame, feature_names: list[str]
) -> shap.Explanation:
    """Computes weighted SHAP values for ensemble models.

    Args:
        model (object): Ensemble model with estimators and weights attributes.
        X (pd.DataFrame): Input data for SHAP computation.
        feature_names (list[str]): Names of the features in X.

    Returns:
        shap.Explanation: Weighted SHAP explanation object for the ensemble.

    Raises:
        ValueError: If no valid models found in ensemble.
    """
    fitted_estimators = model.fitted_estimators_
    n_estimators = len(fitted_estimators)

    max_ensemble_size = 50
    if n_estimators > max_ensemble_size:
        raise ValueError(
            f"Ensemble too large ({n_estimators} estimators). "
            f"Maximum supported: {max_ensemble_size} for SHAP computation."
        )

    weights = _get_ensemble_weights(model)
    shap_values_list = []
    valid_indices = []

    try:
        for idx, estimator in enumerate(fitted_estimators):
            try:
                nested_estimators = getattr(estimator, "fitted_estimators_", None)
                if nested_estimators:
                    if len(nested_estimators) > 10:
                        continue
                    shap_val = _compute_ensemble_shap(estimator, X, feature_names)
                else:
                    shap_val = _compute_single_model_shap(estimator, X)

                shap_values_list.append(shap_val)
                valid_indices.append(idx)
                gc.collect()

            except (ValueError, MemoryError):
                continue

        if not shap_values_list:
            raise ValueError(
                "No valid models found in ensemble (all are HistGradientBoosting or failed)."
            )

        valid_weights = (
            weights[valid_indices]
            if len(weights) > len(valid_indices)
            else weights[: len(shap_values_list)]
        )
        valid_weights = valid_weights / valid_weights.sum()

        first_shap = shap_values_list[0]
        shap_values_weighted = np.zeros_like(first_shap.values, dtype=np.float64)
        base_values_weighted = np.zeros_like(first_shap.base_values, dtype=np.float64)

        for weight, shap_val in zip(valid_weights, shap_values_list):
            shap_values_weighted += weight * shap_val.values
            base_values_weighted += weight * shap_val.base_values

        return shap.Explanation(
            values=shap_values_weighted,
            base_values=base_values_weighted,
            data=X.values,
            feature_names=feature_names,
        )

    finally:
        del shap_values_list
        gc.collect()


def _get_ensemble_weights(model: object) -> np.ndarray:
    """Extract and validate weights from ensemble model.

    Args:
        model (object): Ensemble model with estimators and weights attributes.

    Returns:
        np.ndarray: Normalized weights array.

    Raises:
        ValueError: If no valid estimators found.
    """
    weights = getattr(model, "weights", None)

    if weights is None:
        fitted_estimators = getattr(model, "fitted_estimators_", None)
        if fitted_estimators:
            weights = getattr(fitted_estimators[0], "weights", None)
            if weights is None:
                weights = np.ones(len(fitted_estimators), dtype=np.float64)
        else:
            raise ValueError(
                "Ensemble-like model has no fitted_estimators_ to compute SHAP values."
            )
    else:
        weights = np.asarray(weights, dtype=np.float64)

    if len(weights) == 0:
        raise ValueError("No weights available for ensemble.")

    weights = np.abs(weights)
    weights[~np.isfinite(weights)] = 0.0

    if weights.sum() == 0:
        raise ValueError("All weights are zero or invalid.")

    return weights


if __name__ == "__main__":
    print("shap_values module loaded.")
