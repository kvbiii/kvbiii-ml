import numpy as np
import pandas as pd
from pandas.io.formats.style import Styler
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    explained_variance_score,
    f1_score,
    log_loss,
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_squared_error,
    median_absolute_error,
    precision_recall_fscore_support,
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
                        ("color", "#764ba2"),
                        ("margin-bottom", "20px"),
                        ("letter-spacing", "1.5px"),
                        ("text-shadow", "2px 2px 4px rgba(0,0,0,0.1)"),
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

    def _safe_mape(y_true: pd.Series | np.ndarray, y_pred: np.ndarray) -> float:
        """Computes MAPE on non-zero targets to avoid division by zero."""
        y_true_array = np.asarray(y_true)
        y_pred_array = np.asarray(y_pred)
        non_zero_mask = y_true_array != 0
        if not np.any(non_zero_mask):
            return np.nan
        return float(
            mean_absolute_percentage_error(
                y_true_array[non_zero_mask],
                y_pred_array[non_zero_mask],
            )
        )

    metric_funcs = {
        "MAE": mean_absolute_error,
        "MedAE": median_absolute_error,
        "MSE": mean_squared_error,
        "RMSE": root_mean_squared_error,
        "MAPE": _safe_mape,
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


def _per_class_roc_auc(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    classes: np.ndarray,
) -> np.ndarray:
    """Compute one-vs-rest ROC-AUC for each class, returning nan on failure."""
    proba_matrix = (
        np.column_stack([1 - y_proba, y_proba]) if y_proba.ndim == 1 else y_proba
    )
    results = []
    for i, cls in enumerate(classes):
        y_binary = (y_true == cls).astype(int)
        col_idx = min(i, proba_matrix.shape[1] - 1)
        try:
            results.append(float(roc_auc_score(y_binary, proba_matrix[:, col_idx])))
        except ValueError:
            results.append(np.nan)
    return np.array(results)


def _build_per_class_split(
    y_true: pd.Series | np.ndarray,
    y_pred: np.ndarray,
    y_proba: np.ndarray | None,
    classes: np.ndarray,
    split_label: str,
) -> pd.DataFrame:
    """Build per-class precision/recall/F1/support (and optionally ROC-AUC) for one split."""
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=classes, zero_division=0
    )
    cols: dict[tuple[str, str], np.ndarray] = {
        (split_label, "Precision"): precision,
        (split_label, "Recall"): recall,
        (split_label, "F1"): f1,
        (split_label, "Support"): support.astype(float),
    }
    if y_proba is not None:
        cols[(split_label, "ROC-AUC")] = _per_class_roc_auc(
            np.asarray(y_true), y_proba, classes
        )
    df = pd.DataFrame(cols, index=classes)
    df.columns = pd.MultiIndex.from_tuples(df.columns)
    return df


def _safe_roc_auc(
    y_true: pd.Series | np.ndarray,
    y_proba: np.ndarray | None,
    average: str,
) -> float:
    """Compute ROC-AUC safely, returning nan on failure."""
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
    y_true: pd.Series | np.ndarray,
    y_proba: np.ndarray | None,
) -> float:
    """Compute log loss safely, returning nan on failure."""
    if y_proba is None:
        return np.nan
    try:
        return log_loss(y_true, y_proba)
    except (ValueError, IndexError):
        return np.nan


def _apply_cutoff(
    y_proba: np.ndarray,
    cutoff_val: float | list[float],
) -> np.ndarray:
    """
    Apply probability threshold(s) to produce class predictions.

    Args:
        y_proba (np.ndarray): Predicted probabilities.
        cutoff_val (float | list[float]): Single threshold for binary or per-class
            thresholds for multiclass.

    Returns:
        np.ndarray: Integer class predictions.

    Raises:
        ValueError: If cutoff list length does not match number of classes.
    """
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


def _build_overall_table(
    y_train_true: pd.Series | np.ndarray,
    y_train_pred: np.ndarray,
    y_test_true: pd.Series | np.ndarray,
    y_test_pred: np.ndarray,
    y_train_proba: np.ndarray | None,
    y_test_proba: np.ndarray | None,
    average: str,
    caption: str,
) -> Styler:
    """
    Build and style the overall classification metrics table.

    Args:
        y_train_true (pd.Series | np.ndarray): True training labels.
        y_train_pred (np.ndarray): Predicted training labels.
        y_test_true (pd.Series | np.ndarray): True test labels.
        y_test_pred (np.ndarray): Predicted test labels.
        y_train_proba (np.ndarray | None): Training probabilities for ROC-AUC/Log Loss.
        y_test_proba (np.ndarray | None): Test probabilities for ROC-AUC/Log Loss.
        average (str): Averaging strategy for multi-class metrics.
        caption (str): Table caption.

    Returns:
        Styler: Styled overall metrics table.
    """
    metric_configs = [
        ("Accuracy", accuracy_score, {}),
        ("Balanced Accuracy", balanced_accuracy_score, {}),
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
        _safe_roc_auc(y_train_true, y_train_proba, average),
        _safe_roc_auc(y_test_true, y_test_proba, average),
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

    return _apply_fancy_styling(styled_df, caption, gradient_config)


def _build_per_class_table(
    y_train_true: pd.Series | np.ndarray,
    y_train_pred: np.ndarray,
    y_test_true: pd.Series | np.ndarray,
    y_test_pred: np.ndarray,
    y_train_proba: np.ndarray | None,
    y_test_proba: np.ndarray | None,
    classes: np.ndarray,
    caption: str,
) -> Styler:
    """
    Build and style the per-class classification metrics table.

    Args:
        y_train_true (pd.Series | np.ndarray): True training labels.
        y_train_pred (np.ndarray): Predicted training labels.
        y_test_true (pd.Series | np.ndarray): True test labels.
        y_test_pred (np.ndarray): Predicted test labels.
        y_train_proba (np.ndarray | None): Training probabilities for per-class ROC-AUC.
        y_test_proba (np.ndarray | None): Test probabilities for per-class ROC-AUC.
        classes (np.ndarray): Unique class labels.
        caption (str): Table caption.

    Returns:
        Styler: Styled per-class metrics table.
    """
    train_per_class = _build_per_class_split(
        y_train_true, y_train_pred, y_train_proba, classes, "Train"
    )
    test_per_class = _build_per_class_split(
        y_test_true, y_test_pred, y_test_proba, classes, "Test"
    )

    per_class_df = pd.concat([train_per_class, test_per_class], axis=1)
    per_class_df.index = [f"Class {c}" for c in classes]

    metric_cols = [col for col in per_class_df.columns if col[1] != "Support"]
    support_cols = [col for col in per_class_df.columns if col[1] == "Support"]

    format_dict: dict[tuple[str, str], str] = {col: "{:.4f}" for col in metric_cols}
    format_dict.update({col: "{:.0f}" for col in support_cols})

    per_class_styled = (
        per_class_df.style.format(format_dict)
        .background_gradient(subset=metric_cols, cmap="YlGnBu", vmin=0.0, vmax=1.0)
        .background_gradient(subset=support_cols, cmap="Blues", vmin=0.0, vmax=None)
    )
    return _apply_fancy_styling(per_class_styled, caption, {})


def classification_results(
    y_train_true: pd.Series | np.ndarray,
    y_train_pred: np.ndarray,
    y_test_true: pd.Series | np.ndarray,
    y_test_pred: np.ndarray,
    y_train_proba: np.ndarray | None = None,
    y_test_proba: np.ndarray | None = None,
    id2label: dict[int, str] | None = None,
    cutoffs: float | list[float] | None = None,
    average: str = "weighted",
) -> tuple[Styler, ...]:
    """
    Generate comprehensive styled DataFrames with classification evaluation metrics.

    Always returns an overall metrics table computed with the original predictions
    (default cutoffs, typically 0.5). When `cutoffs` is provided, a second overall table
    is appended using the custom threshold(s) applied to probabilities. For multiclass
    problems (> 2 classes), matching per-class tables are appended in the same order.

    Args:
        y_train_true (pd.Series | np.ndarray): True training target values.
        y_train_pred (np.ndarray): Predicted training target values.
        y_test_true (pd.Series | np.ndarray): True testing target values.
        y_test_pred (np.ndarray): Predicted testing target values.
        y_train_proba (np.ndarray | None, optional): Predicted probabilities for train. Defaults to None.
        y_test_proba (np.ndarray | None, optional): Predicted probabilities for test. Defaults to None.
        id2label (dict[int, str] | None, optional): Mapping from class indices to labels for nicer display. Defaults to None (uses class indices).
        cutoffs (float | list[float] | None, optional): Custom threshold(s). For binary: single float. For multi-class: list of floats per class. Defaults to None.
        average (str, optional): Averaging method for multi-class metrics. Defaults to "weighted".

    Returns:
        tuple[Styler, Styler] | tuple[Styler, Styler, Styler, Styler]:
            When ``cutoffs`` is None (always 2 tables):
                [0] overall_metrics_styled  — overall metrics at default cutoff.
                [1] per_class_metrics_styled — per-class metrics at default cutoff.
            When ``cutoffs`` is provided (always 4 tables):
                [0] overall_metrics_styled         — overall, default cutoff.
                [1] overall_metrics_cutoff_styled  — overall, custom cutoff.
                [2] per_class_metrics_styled        — per-class, default cutoff.
                [3] per_class_metrics_cutoff_styled — per-class, custom cutoff.

    Raises:
        ValueError: If ``cutoffs`` is provided but probabilities or ``id2label`` are missing.
        ValueError: If cutoff list length does not match number of classes.
    """
    if cutoffs is not None and (
        y_train_proba is None or y_test_proba is None or id2label is None
    ):
        raise ValueError("Probabilities and class labels are required for cutoff")

    classes = np.unique(
        np.concatenate([np.asarray(y_train_true), np.asarray(y_test_true)])
    )

    overall_default = _build_overall_table(
        y_train_true,
        y_train_pred,
        y_test_true,
        y_test_pred,
        y_train_proba,
        y_test_proba,
        average,
        "🎯 Classification Performance Metrics - Default Cutoff (0.5)",
    )
    per_class_default = _build_per_class_table(
        y_train_true,
        y_train_pred,
        y_test_true,
        y_test_pred,
        y_train_proba,
        y_test_proba,
        classes,
        "📋 Per-Class Performance Metrics - Default Cutoff (0.5)",
    )

    if cutoffs is None:
        return overall_default, per_class_default

    y_train_pred_cutoff = pd.Series(_apply_cutoff(y_train_proba, cutoffs)).map(id2label)
    y_test_pred_cutoff = pd.Series(_apply_cutoff(y_test_proba, cutoffs)).map(id2label)
    cutoff_label = str(cutoffs)

    overall_cutoff = _build_overall_table(
        y_train_true,
        y_train_pred_cutoff,
        y_test_true,
        y_test_pred_cutoff,
        y_train_proba,
        y_test_proba,
        average,
        f"🎯 Classification Performance Metrics - Cutoff: {cutoff_label}",
    )
    per_class_cutoff = _build_per_class_table(
        y_train_true,
        y_train_pred_cutoff,
        y_test_true,
        y_test_pred_cutoff,
        y_train_proba,
        y_test_proba,
        classes,
        f"📋 Per-Class Performance Metrics - Cutoff: {cutoff_label}",
    )
    return overall_default, overall_cutoff, per_class_default, per_class_cutoff
