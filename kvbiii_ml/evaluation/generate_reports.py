import pandas as pd
import numpy as np
from sklearn.metrics import (
    mean_absolute_error,
    median_absolute_error,
    mean_squared_error,
    root_mean_squared_error,
    mean_absolute_percentage_error,
    r2_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    explained_variance_score,
)

try:  # Optional dependency for plotting
    import matplotlib.pyplot as plt  # type: ignore
except Exception:  # pragma: no cover - allow running without matplotlib
    plt = None  # type: ignore
from sklearn.metrics import roc_curve, auc


def regression_results(
    y_train_true: pd.Series | np.ndarray,
    y_train_pred: np.ndarray,
    y_test_true: pd.Series | np.ndarray,
    y_test_pred: np.ndarray,
) -> pd.DataFrame:
    """
    Generates a comprehensive styled DataFrame with regression evaluation metrics.

    Args:
        y_train_true (pd.Series | np.ndarray): True training target values.
        y_train_pred (np.ndarray): Predicted training target values.
        y_test_true (pd.Series | np.ndarray): True testing target values.
        y_test_pred (np.ndarray): Predicted testing target values.

    Returns:
        pd.DataFrame: A styled DataFrame with comprehensive regression metrics.
    """

    def safe_mape(y_true, y_pred):
        """Calculate MAPE safely, handling zero values."""
        try:
            return mean_absolute_percentage_error(y_true, y_pred)
        except ValueError:
            return np.nan

    metrics = {
        "MAE": [
            mean_absolute_error(y_train_true, y_train_pred),
            mean_absolute_error(y_test_true, y_test_pred),
        ],
        "MedAE": [
            median_absolute_error(y_train_true, y_train_pred),
            median_absolute_error(y_test_true, y_test_pred),
        ],
        "MSE": [
            mean_squared_error(y_train_true, y_train_pred),
            mean_squared_error(y_test_true, y_test_pred),
        ],
        "RMSE": [
            root_mean_squared_error(y_train_true, y_train_pred),
            root_mean_squared_error(y_test_true, y_test_pred),
        ],
        "MAPE": [
            safe_mape(y_train_true, y_train_pred),
            safe_mape(y_test_true, y_test_pred),
        ],
        "R²": [
            r2_score(y_train_true, y_train_pred),
            r2_score(y_test_true, y_test_pred),
        ],
    }

    df = pd.DataFrame(metrics, index=["Train", "Test"])

    styled_df = (
        df.style.set_caption("📊 Regression Performance Metrics")
        .format(
            {
                "MAE": "{:.4f}",
                "MedAE": "{:.4f}",
                "MSE": "{:.4f}",
                "RMSE": "{:.4f}",
                "MAPE": "{:.2%}",
                "R²": "{:.4f}",
            }
        )
        # Color coding
        .background_gradient(subset=["R²"], cmap="RdYlGn", vmin=0, vmax=1)
        .background_gradient(subset=["MAE", "MedAE", "MSE", "RMSE"], cmap="RdYlGn_r")
        # Table-wide font and styling
        .set_properties(
            **{
                "text-align": "left",
                "font-family": "Times New Roman",
                "font-size": "1.5em",
                "background-color": "#f9f9f9",
                "border": "3px solid #ddd",
                "color": "#333",
            }
        )
        # Table and caption styles
        .set_table_styles(
            [
                {
                    "selector": "th",
                    "props": [
                        ("font-weight", "bold"),
                        ("border", "3px solid #ddd"),
                        ("background-color", "#4a90e2"),
                        ("color", "white"),
                        ("text-align", "center"),
                        ("font-size", "1.4em"),  # Bigger header font
                        ("padding", "10px"),
                    ],
                },
                {
                    "selector": "caption",
                    "props": [
                        ("font-size", "1.6em"),  # Bigger caption font
                        ("font-weight", "bold"),
                        ("color", "#4a90e2"),
                        ("margin-bottom", "15px"),
                    ],
                },
            ]
        )
    )
    return styled_df


def classification_results(
    y_train_true: pd.Series | np.ndarray,
    y_train_pred: np.ndarray,
    y_test_true: pd.Series | np.ndarray,
    y_test_pred: np.ndarray,
    y_train_proba: np.ndarray | None = None,
    y_test_proba: np.ndarray | None = None,
    average: str = "weighted",
    id2label: dict[int, str] | None = None,
    cutoff: float | list[float] | None = None,
) -> pd.DataFrame:
    """
    Generates a comprehensive styled DataFrame with classification evaluation metrics.

    Args:
        cutoff (float | list[float] | None): Custom threshold(s) for classification.
            - For binary: single float (e.g., 0.7)
            - For multi-class: list of floats, one per class (e.g., [0.3, 0.4, 0.3])
            If None, uses the original predictions.
    """

    def apply_cutoff(
        y_proba: np.ndarray, cutoff_value: float | list[float]
    ) -> np.ndarray:
        """Apply cutoff threshold(s) to probabilities to get predictions."""
        if y_proba is None:
            raise ValueError("Probabilities must be provided when using cutoff")

        if isinstance(cutoff_value, (int, float)):
            if y_proba.ndim == 1:
                return (y_proba >= cutoff_value).astype(int)
            else:
                return (y_proba[:, 1] >= cutoff_value).astype(int)
        else:
            cutoff_array = np.array(cutoff_value)
            if y_proba.shape[1] != len(cutoff_array):
                raise ValueError(
                    f"Number of cutoffs ({len(cutoff_array)}) must match number of classes ({y_proba.shape[1]})"
                )
            normalized = y_proba / cutoff_array
            return np.argmax(normalized, axis=1)

    if cutoff is not None:
        if y_train_proba is None or y_test_proba is None:
            raise ValueError(
                "Probabilities (y_train_proba and y_test_proba) must be provided when using cutoff"
            )
        y_train_pred = apply_cutoff(y_train_proba, cutoff)
        y_test_pred = apply_cutoff(y_test_proba, cutoff)

    def safe_roc_auc(
        y_true: pd.Series | np.ndarray, y_proba: np.ndarray, multi_class: str = "ovr"
    ) -> float:
        if y_proba is None:
            return np.nan
        try:
            if len(np.unique(y_true)) == 2:
                return roc_auc_score(
                    y_true, y_proba[:, 1] if y_proba.ndim > 1 else y_proba
                )
            else:
                return roc_auc_score(
                    y_true, y_proba, multi_class=multi_class, average=average
                )
        except (ValueError, IndexError):
            return np.nan

    metrics = {
        "Accuracy": [
            accuracy_score(y_train_true, y_train_pred),
            accuracy_score(y_test_true, y_test_pred),
        ],
        "Precision": [
            precision_score(
                y_train_true, y_train_pred, average=average, zero_division=0
            ),
            precision_score(y_test_true, y_test_pred, average=average, zero_division=0),
        ],
        "Recall": [
            recall_score(y_train_true, y_train_pred, average=average, zero_division=0),
            recall_score(y_test_true, y_test_pred, average=average, zero_division=0),
        ],
        "F1-Score (weighted)": [
            f1_score(y_train_true, y_train_pred, average=average, zero_division=0),
            f1_score(y_test_true, y_test_pred, average=average, zero_division=0),
        ],
        "F1-Score (macro)": [
            f1_score(y_train_true, y_train_pred, average="macro", zero_division=0),
            f1_score(y_test_true, y_test_pred, average="macro", zero_division=0),
        ],
        "ROC-AUC": [
            safe_roc_auc(y_train_true, y_train_proba),
            safe_roc_auc(y_test_true, y_test_proba),
        ],
    }

    df = pd.DataFrame(metrics, index=["Train", "Test"])

    styled_df = (
        df.style.set_caption("🎯 Classification Performance Metrics")
        .format({col: "{:.4f}" for col in df.columns})
        .background_gradient(cmap="YlGnBu", vmin=0, vmax=1)
        .set_properties(
            **{
                "text-align": "left",
                "font-family": "Times New Roman",
                "font-size": "1.5em",
                "background-color": "#f9f9f9",
                "border": "3px solid #ddd",
                "color": "#333",
            }
        )
        .set_table_styles(
            [
                {
                    "selector": "th",
                    "props": [
                        ("font-weight", "bold"),
                        ("border", "3px solid #ddd"),
                        ("background-color", "#4a90e2"),
                        ("color", "white"),
                        ("text-align", "center"),
                        ("font-size", "1.4em"),  # Bigger header font
                        ("padding", "10px"),
                    ],
                },
                {
                    "selector": "caption",
                    "props": [
                        ("font-size", "1.6em"),  # Bigger caption font
                        ("font-weight", "bold"),
                        ("color", "#4a90e2"),
                        ("margin-bottom", "15px"),
                    ],
                },
            ]
        )
    )
    return styled_df


if __name__ == "__main__":
    from sklearn.datasets import make_regression, make_classification
    from sklearn.model_selection import train_test_split
    from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
    from sklearn.linear_model import LogisticRegression

    print("🔍 Testing Regression Results Function:")
    print("=" * 50)

    X_reg, y_reg = make_regression(
        n_samples=1000, n_features=10, noise=0.1, random_state=42
    )
    X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(
        X_reg, y_reg, test_size=0.2, random_state=42
    )

    reg_model = RandomForestRegressor(n_estimators=100, random_state=42)
    reg_model.fit(X_train_reg, y_train_reg)

    y_train_pred_reg = reg_model.predict(X_train_reg)
    y_test_pred_reg = reg_model.predict(X_test_reg)

    regression_report = regression_results(
        y_train_reg, y_train_pred_reg, y_test_reg, y_test_pred_reg
    )
    print(regression_report.to_string())

    print("\n🎯 Testing Classification Results Function:")
    print("=" * 50)

    X_clf, y_clf = make_classification(
        n_samples=1000, n_features=10, n_classes=3, n_informative=8, random_state=42
    )
    X_train_clf, X_test_clf, y_train_clf, y_test_clf = train_test_split(
        X_clf, y_clf, test_size=0.2, random_state=42
    )

    clf_model = LogisticRegression(random_state=42, max_iter=1000)
    clf_model.fit(X_train_clf, y_train_clf)

    y_train_pred_clf = clf_model.predict(X_train_clf)
    y_test_pred_clf = clf_model.predict(X_test_clf)

    y_train_proba_clf = clf_model.predict_proba(X_train_clf)
    y_test_proba_clf = clf_model.predict_proba(X_test_clf)

    classification_report = classification_results(
        y_train_clf,
        y_train_pred_clf,
        y_test_clf,
        y_test_pred_clf,
        y_train_proba_clf,
        y_test_proba_clf,
    )
    print(classification_report.to_string())
