import pandas as pd
import numpy as np
import shap
import gc


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
    if feature_names is None:
        feature_names = list(X.columns)

    is_ensemble = (
        hasattr(model, "fitted_estimators_")
        and getattr(model, "fitted_estimators_", None) is not None
    )

    if not is_ensemble:
        return _compute_single_model_shap(model, X)

    return _compute_ensemble_shap(model, X, feature_names)


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

    try:
        explainer = shap.TreeExplainer(
            model, feature_perturbation="tree_path_dependent"
        )
        try:
            result = explainer(X)
        except AttributeError:
            result = explainer(X.values)
        return result
    finally:
        gc.collect()


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
    weights = _get_ensemble_weights(model)

    max_ensemble_size = 50
    if len(getattr(model, "fitted_estimators_", [])) > max_ensemble_size:
        raise ValueError(
            f"Ensemble too large ({len(model.fitted_estimators_)} estimators). "
            f"Maximum supported: {max_ensemble_size} for SHAP computation."
        )

    valid_indices = []
    shap_values_list = []

    try:
        for idx, estimator in enumerate(model.fitted_estimators_):
            try:
                if hasattr(estimator, "fitted_estimators_") and getattr(
                    estimator, "fitted_estimators_", None
                ):
                    if len(estimator.fitted_estimators_) > 10:
                        continue
                    shap_val = _compute_ensemble_shap(estimator, X, feature_names)
                else:
                    shap_val = _compute_single_model_shap(estimator, X)

                shap_values_list.append(shap_val)
                valid_indices.append(idx)

                gc.collect()

            except (ValueError, MemoryError) as e:
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
            try:
                np.add(
                    shap_values_weighted,
                    weight * shap_val.values,
                    out=shap_values_weighted,
                )
                np.add(
                    base_values_weighted,
                    weight * shap_val.base_values,
                    out=base_values_weighted,
                )
            except Exception as e:
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
    if hasattr(model, "weights") and getattr(model, "weights", None) is not None:
        weights = np.array(model.weights, dtype=np.float64)
    elif getattr(model, "fitted_estimators_", None) and hasattr(
        model.fitted_estimators_[0], "weights"
    ):
        weights = np.array(model.fitted_estimators_[0].weights, dtype=np.float64)
    else:
        n_est = len(getattr(model, "fitted_estimators_", []) or [])
        if n_est == 0:
            raise ValueError(
                "Ensemble-like model has no fitted_estimators_ to compute SHAP values."
            )
        weights = np.ones(n_est, dtype=np.float64)

    if len(weights) == 0:
        raise ValueError("No weights available for ensemble.")

    weights = np.abs(weights)
    weights[~np.isfinite(weights)] = 0.0

    if weights.sum() == 0:
        raise ValueError("All weights are zero or invalid.")

    return weights


if __name__ == "__main__":
    from sklearn.datasets import make_regression, make_classification
    from sklearn.model_selection import train_test_split
    from sklearn.ensemble import (
        RandomForestRegressor,
        RandomForestClassifier,
        ExtraTreesClassifier,
    )
    from sklearn.tree import DecisionTreeClassifier
    from kvbiii_ml.modeling.training.ensemble_model import EnsembleModel
    from kvbiii_ml.modeling.training.cross_validation import CrossValidationTrainer
    from sklearn.metrics import r2_score, accuracy_score

    print("================ SHAP EXAMPLES ================")

    # 1. Single LGBMRegressor (regression)
    print("\n[1] Single LGBMRegressor (regression)")
    try:
        from lightgbm import LGBMRegressor

        X_reg, y_reg = make_regression(
            n_samples=300, n_features=10, n_informative=6, noise=0.2, random_state=42
        )
        feature_names_reg = [f"f{i}" for i in range(X_reg.shape[1])]
        X_reg_df = pd.DataFrame(X_reg, columns=feature_names_reg)
        X_train, X_test, y_train, y_test = train_test_split(
            X_reg_df, y_reg, test_size=0.25, random_state=17
        )
        lgbm = LGBMRegressor(random_state=42, n_estimators=150, verbose=-1)
        lgbm.fit(X_train, y_train)
        shap_exp_lgbm = compute_shap_values(lgbm, X_test.head(20))
        print("SHAP values shape:", shap_exp_lgbm.values.shape)
        print("R2 sample:", r2_score(y_test[:20], lgbm.predict(X_test.head(20))))
    except ImportError:
        print(
            "LightGBM not installed; skipping this example. Install lightgbm to enable."
        )

    # 2. EnsembleModel with classifiers
    print("\n[2] EnsembleModel with tree classifiers (classification)")
    X_clf, y_clf = make_classification(
        n_samples=400,
        n_features=8,
        n_informative=5,
        n_redundant=1,
        n_classes=2,
        random_state=42,
    )
    feature_names_clf = [f"c{i}" for i in range(X_clf.shape[1])]
    X_clf_df = pd.DataFrame(X_clf, columns=feature_names_clf)
    Xc_train, Xc_test, yc_train, yc_test = train_test_split(
        X_clf_df, y_clf, test_size=0.25, random_state=17
    )
    est1 = RandomForestClassifier(n_estimators=120, max_depth=5, random_state=1)
    est2 = ExtraTreesClassifier(n_estimators=150, max_depth=6, random_state=2)
    est3 = DecisionTreeClassifier(max_depth=5, random_state=3)
    ensemble_cls = EnsembleModel([est1, est2, est3], problem_type="classification")
    ensemble_cls.fit(Xc_train, yc_train)
    shap_exp_ensemble = compute_shap_values(ensemble_cls, Xc_test.head(15))
    print("SHAP values shape (ensemble classifiers):", shap_exp_ensemble.values.shape)
    y_pred_ens = ensemble_cls.predict(Xc_test)
    print("Accuracy:", accuracy_score(yc_test, y_pred_ens))

    # 3. CrossValidationTrainer with single estimator
    print("\n[3] CrossValidationTrainer with single RandomForestRegressor")
    X_reg2, y_reg2 = make_regression(
        n_samples=250, n_features=12, n_informative=7, noise=0.3, random_state=11
    )
    feature_names_reg2 = [f"r{i}" for i in range(X_reg2.shape[1])]
    X_reg2_df = pd.DataFrame(X_reg2, columns=feature_names_reg2)
    Xr_train, Xr_test, yr_train, yr_test = train_test_split(
        X_reg2_df, y_reg2, test_size=0.3, random_state=19
    )
    rf_reg = RandomForestRegressor(n_estimators=80, max_depth=6, random_state=4)
    cv_trainer_reg = CrossValidationTrainer(
        metric_name="R2", problem_type="regression", verbose=False
    )
    cv_trainer_reg.fit(rf_reg, Xr_train, pd.Series(yr_train))
    shap_exp_cv_single = compute_shap_values(cv_trainer_reg, Xr_test.head(10))
    print("SHAP values shape (CV single regressor):", shap_exp_cv_single.values.shape)

    # 4. CrossValidationTrainer with EnsembleModel (classification)
    print("\n[4] CrossValidationTrainer with EnsembleModel (classification)")
    X_clf2, y_clf2 = make_classification(
        n_samples=300,
        n_features=9,
        n_informative=6,
        n_redundant=1,
        n_classes=2,
        random_state=21,
    )
    feature_names_clf2 = [f"ec{i}" for i in range(X_clf2.shape[1])]
    X_clf2_df = pd.DataFrame(X_clf2, columns=feature_names_clf2)
    Xc2_train, Xc2_test, yc2_train, yc2_test = train_test_split(
        X_clf2_df, y_clf2, test_size=0.3, random_state=29
    )
    base1 = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=5)
    base2 = ExtraTreesClassifier(n_estimators=120, max_depth=6, random_state=6)
    nested_ensemble = EnsembleModel([base1, base2], problem_type="classification")
    cv_trainer_cls = CrossValidationTrainer(
        metric_name="Accuracy", problem_type="classification", verbose=False
    )
    cv_trainer_cls.fit(nested_ensemble, Xc2_train, pd.Series(yc2_train))
    shap_exp_cv_ens = compute_shap_values(cv_trainer_cls, Xc2_test.head(12))
    print(
        "SHAP values shape (CV with ensemble classifier):",
        shap_exp_cv_ens.values.shape,
    )
    X_clf2, y_clf2 = make_classification(
        n_samples=300,
        n_features=9,
        n_informative=6,
        n_redundant=1,
        n_classes=2,
        random_state=21,
    )
    feature_names_clf2 = [f"ec{i}" for i in range(X_clf2.shape[1])]
    X_clf2_df = pd.DataFrame(X_clf2, columns=feature_names_clf2)
    Xc2_train, Xc2_test, yc2_train, yc2_test = train_test_split(
        X_clf2_df, y_clf2, test_size=0.3, random_state=29
    )
    base1 = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=5)
    base2 = ExtraTreesClassifier(n_estimators=120, max_depth=6, random_state=6)
    nested_ensemble = EnsembleModel([base1, base2], problem_type="classification")
    cv_trainer_cls = CrossValidationTrainer(
        metric_name="Accuracy", problem_type="classification", verbose=False
    )
    cv_trainer_cls.fit(nested_ensemble, Xc2_train, pd.Series(yc2_train))
    shap_exp_cv_ens = compute_shap_values(cv_trainer_cls, Xc2_test.head(12))
    print(
        "SHAP values shape (CV with ensemble classifier):",
        shap_exp_cv_ens.values.shape,
    )
