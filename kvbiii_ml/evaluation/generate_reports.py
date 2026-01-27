"""
Evaluation report generation for regression and classification models.

This module provides functions to generate styled performance metric reports
for machine learning models, including comprehensive metrics and visualizations.
"""

import numpy as np
import pandas as pd
from pandas.io.formats.style import Styler
from sklearn.metrics import (
    accuracy_score,
    explained_variance_score,
    f1_score,
    log_loss,
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_squared_error,
    median_absolute_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
    root_mean_squared_error,
)


def _apply_fancy_styling(
    styled_df: Styler,
    caption: str,
    gradient_cols: dict[tuple[str, ...], tuple[str, float | None, float | None]],
) -> Styler:
    """
    Apply consistent fancy styling to DataFrame.

    Args:
        styled_df (Styler): Styled DataFrame to modify.
        caption (str): Caption for the table.
        gradient_cols (dict[tuple[str, ...], tuple[str, float | None, float | None]]):
            Column gradient configs with cmap, vmin, vmax.

    Returns:
        Styler: Enhanced styled DataFrame.
    """
    for cols, (cmap, vmin, vmax) in gradient_cols.items():
        styled_df = styled_df.background_gradient(
            subset=list(cols),
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
        )

    return (
        styled_df.set_caption(caption)
        .set_properties(
            **{
                "text-align": "center",
                "font-family": "Segoe UI, Arial, sans-serif",
                "font-size": "1.1em",
                "background-color": "#ffffff",
                "border": "2px solid #e0e0e0",
                "color": "#2c3e50",
                "padding": "12px",
            }
        )
        .set_table_styles(
            [
                {
                    "selector": "th",
                    "props": [
                        ("font-weight", "700"),
                        ("border", "2px solid #bdc3c7"),
                        (
                            "background",
                            "linear-gradient(135deg, #667eea 0%, #764ba2 100%)",
                        ),
                        ("color", "#ffffff"),
                        ("text-align", "center"),
                        ("font-size", "1.15em"),
                        ("padding", "14px"),
                        ("text-transform", "uppercase"),
                        ("letter-spacing", "1px"),
                    ],
                },
                {
                    "selector": "caption",
                    "props": [
                        ("font-size", "1.8em"),
                        ("font-weight", "800"),
                        (
                            "background",
                            "linear-gradient(135deg, #f093fb 0%, #f5576c 100%)",
                        ),
                        ("-webkit-background-clip", "text"),
                        ("-webkit-text-fill-color", "transparent"),
                        ("margin-bottom", "20px"),
                        ("letter-spacing", "1.5px"),
                    ],
                },
                {
                    "selector": "tbody tr:hover",
                    "props": [
                        ("background-color", "#f8f9fa"),
                        ("transform", "scale(1.01)"),
                        ("transition", "all 0.2s ease"),
                    ],
                },
                {
                    "selector": "",
                    "props": [
                        ("border-collapse", "separate"),
                        ("border-spacing", "0"),
                        ("box-shadow", "0 4px 6px rgba(0,0,0,0.1)"),
                        ("border-radius", "8px"),
                        ("overflow", "hidden"),
                    ],
                },
            ]
        )
    )


def regression_results(
    y_train_true: pd.Series | np.ndarray,
    y_train_pred: np.ndarray,
    y_test_true: pd.Series | np.ndarray,
    y_test_pred: np.ndarray,
) -> Styler:
    """
    Generate comprehensive styled DataFrame with regression evaluation metrics.

    Args:
        y_train_true (pd.Series | np.ndarray): True training target values.
        y_train_pred (np.ndarray): Predicted training target values.
        y_test_true (pd.Series | np.ndarray): True testing target values.
        y_test_pred (np.ndarray): Predicted testing target values.

    Returns:
        Styler: Styled DataFrame with regression metrics.
    """
    metric_funcs = {
        "MAE": mean_absolute_error,
        "MedAE": median_absolute_error,
        "MSE": mean_squared_error,
        "RMSE": root_mean_squared_error,
        "MAPE": lambda y_t, y_p: (
            mean_absolute_percentage_error(y_t, y_p) if not np.any(y_t == 0) else np.nan
        ),
        "R²": r2_score,
        "Explained Var": explained_variance_score,
    }

    metrics = {
        name: [func(y_train_true, y_train_pred), func(y_test_true, y_test_pred)]
        for name, func in metric_funcs.items()
    }

    df = pd.DataFrame(metrics, index=["Train", "Test"])

    format_dict = {col: "{:.4f}" for col in df.columns}
    format_dict["MAPE"] = "{:.2%}"

    styled_df = df.style.format(format_dict)

    gradient_config = {
        ("R²", "Explained Var"): ("RdYlGn", 0.0, 1.0),
        ("MAE", "MedAE", "MSE", "RMSE"): ("RdYlGn_r", None, None),
    }

    return _apply_fancy_styling(
        styled_df, "📊 Regression Performance Metrics", gradient_config
    )


def classification_results(
    y_train_true: pd.Series | np.ndarray,
    y_train_pred: np.ndarray,
    y_test_true: pd.Series | np.ndarray,
    y_test_pred: np.ndarray,
    y_train_proba: np.ndarray | None = None,
    y_test_proba: np.ndarray | None = None,
    average: str = "weighted",
    cutoff: float | list[float] | None = None,
) -> Styler:
    """
    Generate comprehensive styled DataFrame with classification evaluation metrics.

    Args:
        y_train_true (pd.Series | np.ndarray): True training target values.
        y_train_pred (np.ndarray): Predicted training target values.
        y_test_true (pd.Series | np.ndarray): True testing target values.
        y_test_pred (np.ndarray): Predicted testing target values.
        y_train_proba (np.ndarray | None, optional): Predicted probabilities for train.
            Defaults to None.
        y_test_proba (np.ndarray | None, optional): Predicted probabilities for test.
            Defaults to None.
        average (str, optional): Averaging method for multi-class. Defaults to "weighted".
        cutoff (float | list[float] | None, optional): Custom threshold(s). For binary:
            single float. For multi-class: list per class. Defaults to None.

    Returns:
        Styler: Styled DataFrame with classification metrics.

    Raises:
        ValueError: If probabilities required for cutoff but not provided.
        ValueError: If cutoff length doesn't match number of classes.
    """

    def _apply_cutoff(
        y_proba: np.ndarray, cutoff_val: float | list[float]
    ) -> np.ndarray:
        if y_proba is None:
            raise ValueError("Probabilities required when using cutoff")

        if isinstance(cutoff_val, (int, float)):
            return (
                (y_proba >= cutoff_val).astype(int)
                if y_proba.ndim == 1
                else (y_proba[:, 1] >= cutoff_val).astype(int)
            )

        cutoff_array = np.array(cutoff_val)
        if y_proba.shape[1] != len(cutoff_array):
            raise ValueError(
                f"Cutoff count ({len(cutoff_array)}) must match classes "
                f"({y_proba.shape[1]})"
            )
        return np.argmax(y_proba / cutoff_array, axis=1)

    if cutoff is not None:
        if y_train_proba is None or y_test_proba is None:
            raise ValueError("Probabilities required for cutoff")
        y_train_pred = _apply_cutoff(y_train_proba, cutoff)
        y_test_pred = _apply_cutoff(y_test_proba, cutoff)

    def _safe_roc_auc(
        y_true: pd.Series | np.ndarray, y_proba: np.ndarray | None
    ) -> float:
        if y_proba is None:
            return np.nan
        try:
            return (
                roc_auc_score(y_true, y_proba[:, 1] if y_proba.ndim > 1 else y_proba)
                if len(np.unique(y_true)) == 2
                else roc_auc_score(y_true, y_proba, multi_class="ovr", average=average)
            )
        except (ValueError, IndexError):
            return np.nan

    def _safe_log_loss(
        y_true: pd.Series | np.ndarray, y_proba: np.ndarray | None
    ) -> float:
        if y_proba is None:
            return np.nan
        try:
            return log_loss(y_true, y_proba)
        except (ValueError, IndexError):
            return np.nan

    metric_configs = [
        ("Accuracy", accuracy_score, {}),
        ("Precision", precision_score, {"average": average, "zero_division": 0}),
        ("Recall", recall_score, {"average": average, "zero_division": 0}),
        ("F1 (weighted)", f1_score, {"average": average, "zero_division": 0}),
        ("F1 (macro)", f1_score, {"average": "macro", "zero_division": 0}),
    ]

    metrics = {
        name: [
            func(y_train_true, y_train_pred, **kwargs),
            func(y_test_true, y_test_pred, **kwargs),
        ]
        for name, func, kwargs in metric_configs
    }

    metrics["ROC-AUC"] = [
        _safe_roc_auc(y_train_true, y_train_proba),
        _safe_roc_auc(y_test_true, y_test_proba),
    ]
    metrics["Log Loss"] = [
        _safe_log_loss(y_train_true, y_train_proba),
        _safe_log_loss(y_test_true, y_test_proba),
    ]

    df = pd.DataFrame(metrics, index=["Train", "Test"])

    styled_df = df.style.format({col: "{:.4f}" for col in df.columns})

    gradient_config = {
        tuple(col for col in df.columns if col != "Log Loss"): ("YlGnBu", 0.0, 1.0),
        ("Log Loss",): ("YlOrRd_r", None, None),
    }

    return _apply_fancy_styling(
        styled_df, "🎯 Classification Performance Metrics", gradient_config
    )


if __name__ == "__main__":
    from sklearn.datasets import make_classification, make_regression
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import train_test_split

    print("🔍 Testing Regression Results:\n" + "=" * 50)

    X_reg, y_reg = make_regression(
        n_samples=1000, n_features=10, noise=0.1, random_state=42
    )
    X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(
        X_reg, y_reg, test_size=0.2, random_state=42
    )

    reg_model = RandomForestRegressor(n_estimators=100, random_state=42)
    reg_model.fit(X_train_reg, y_train_reg)

    print(
        regression_results(
            y_train_reg,
            reg_model.predict(X_train_reg),
            y_test_reg,
            reg_model.predict(X_test_reg),
        ).to_string()
    )

    print("\n🎯 Testing Classification Results:\n" + "=" * 50)

    X_clf, y_clf = make_classification(
        n_samples=1000, n_features=10, n_classes=3, n_informative=8, random_state=42
    )
    X_train_clf, X_test_clf, y_train_clf, y_test_clf = train_test_split(
        X_clf, y_clf, test_size=0.2, random_state=42
    )

    clf_model = LogisticRegression(random_state=42, max_iter=1000)
    clf_model.fit(X_train_clf, y_train_clf)

    print(
        classification_results(
            y_train_clf,
            clf_model.predict(X_train_clf),
            y_test_clf,
            clf_model.predict(X_test_clf),
            clf_model.predict_proba(X_train_clf),
            clf_model.predict_proba(X_test_clf),
        ).to_string()
    )
