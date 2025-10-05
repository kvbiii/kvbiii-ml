from __future__ import annotations

import numpy as np
import pandas as pd
from IPython.display import display


def get_top_regression_under_over_errors(
    y_true: pd.Series,
    y_pred: np.ndarray | pd.Series,
    top_n: int = 15,
    log: bool = False,
    epsilon: float = 1e-9,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compute top under- and over-estimated observations.

    Percentage error is (pred - true) / true * 100. Very small or zero true
    values are protected by ``epsilon`` to avoid division by zero. When
    ``log`` is True, y_true and y_pred are treated as log-scaled values and
    additional exponentiated columns are included.

    Args:
        y_true (pd.Series): True target values (index-aligned).
        y_pred (np.ndarray | pd.Series): Predicted values.
        top_n (int, optional): Number of rows to return in each result. Defaults to 15.
        log (bool, optional): Whether inputs are in log-space. Defaults to False.
        epsilon (float, optional): Small value to guard division by zero. Defaults to 1e-9.

    Returns:
        tuple[pd.DataFrame, pd.DataFrame]: (underestimated_df, overestimated_df) DataFrames.
    """
    if not isinstance(y_true, pd.Series):
        raise TypeError("y_true must be a pandas Series aligned to predictions.")

    y_pred = pd.Series(y_pred, index=y_true.index, name="y_pred")

    denom = y_true.replace(0, np.nan)
    denom = denom.fillna(epsilon)
    pct_error = (y_pred - y_true) / denom * 100.0

    # Underestimated -> negative pct_error (prediction < true)
    underestimated = pct_error[pct_error < 0].sort_values().head(top_n)
    # Overestimated -> positive pct_error (prediction > true)
    overestimated = pct_error[pct_error > 0].sort_values(ascending=False).head(top_n)

    def _build(df_index):
        df = pd.DataFrame(
            {
                "True Value": y_true.loc[df_index],
                "Predicted Value": y_pred.loc[df_index],
                "Error": (y_pred - y_true).loc[df_index],
                "Percentage Error (%)": pct_error.loc[df_index],
            }
        )
        if log:
            true_exp = np.exp(df["True Value"])
            pred_exp = np.exp(df["Predicted Value"])
            df["True Value (exp)"] = true_exp
            df["Predicted Value (exp)"] = pred_exp
            df["Error (exp)"] = pred_exp - true_exp
            df["Percentage Error (exp) (%)"] = (df["Error (exp)"] / true_exp) * 100.0
        return df

    return _build(underestimated.index), _build(overestimated.index)


def display_regression_under_over_errors(
    underestimated_df: pd.DataFrame, overestimated_df: pd.DataFrame
) -> None:
    """Display styled tables of under- and over-estimated observations.

    Args:
        underestimated_df (pd.DataFrame): Output from ``get_top_under_overestimated`` for underestimated rows.
        overestimated_df (pd.DataFrame): Output from ``get_top_under_overestimated`` for overestimated rows.

    Returns:
        None: This function renders styled tables (best viewed in notebooks).
    """

    def _style(df: pd.DataFrame, title: str):
        fmt: dict[str, str] = {}
        for col in df.columns:
            if df[col].dtype.kind in "fc":
                if "Percentage Error" in col:
                    fmt[col] = "{:+.2f}%"
                else:
                    fmt[col] = "{:.4f}"
        styler = df.style.set_caption(title).format(fmt)
        error_cols = [c for c in df.columns if "Error" in c]
        if error_cols:
            styler = styler.background_gradient(subset=error_cols, cmap="RdYlGn_r")
        if "Percentage Error (%)" in df.columns:
            styler = styler.bar(
                subset=["Percentage Error (%)"],
                align="mid",
                color=["#d73027", "#1a9850"],
            )
        styler = styler.set_properties(
            **{
                "text-align": "left",
                "font-family": "Times New Roman",
                "font-size": "1.25em",
                "background-color": "#f9f9f9",
                "border": "3px solid #ddd",
                "color": "#333",
            }
        ).set_table_styles(
            [
                {
                    "selector": "th",
                    "props": [
                        ("font-weight", "bold"),
                        ("border", "3px solid #ddd"),
                        ("background-color", "#4a90e2"),
                        ("color", "white"),
                        ("text-align", "center"),
                        ("font-size", "1.2em"),
                        ("padding", "8px"),
                    ],
                },
                {
                    "selector": "caption",
                    "props": [
                        ("font-size", "1.4em"),
                        ("font-weight", "bold"),
                        ("color", "#4a90e2"),
                        ("margin-bottom", "12px"),
                    ],
                },
            ]
        )
        return styler

    display(
        _style(
            underestimated_df,
            f"📉 Top {len(underestimated_df)} Underestimated (by % error)",
        )
    )
    display(
        _style(
            overestimated_df,
            f"📈 Top {len(overestimated_df)} Overestimated (by % error)",
        )
    )


def get_top_classification_errors(
    y_true: pd.Series,
    y_pred_proba: np.ndarray,
    class_id: int,
    id2label: dict[int, str],
    top_n: int = 15,
) -> pd.DataFrame:
    """
    Get top classification errors for a specific class based on biggest probability difference.

    The function always returns errors (false positives and false negatives)
    sorted by the magnitude of the difference between the model's predicted
    class probability and the true class probability (higher = more confident error).

    Args:
        y_true (pd.Series): True class labels (index-aligned).
        y_pred_proba (np.ndarray): Predicted probabilities array (n_samples, n_classes).
        class_id (int): Target class ID to analyze errors for.
        id2label (dict[int, str]): Mapping from class index to class name.
        top_n (int, optional): Number of rows to return. Defaults to 15.

    Returns:
        pd.DataFrame: Top classification errors sorted by confidence/error magnitude.
    """
    if not isinstance(y_true, pd.Series):
        raise TypeError("y_true must be a pandas Series aligned to predictions.")
    if y_pred_proba.shape[0] != len(y_true):
        raise ValueError("y_pred_proba must have the same number of samples as y_true")

    if isinstance(y_true.iloc[0], str):
        label2id = {v: k for k, v in id2label.items()}
        y_true_indices = y_true.map(label2id)
    else:
        y_true_indices = y_true
    if y_pred_proba.ndim == 1:
        y_pred_proba = np.vstack([1 - y_pred_proba, y_pred_proba]).T
    if class_id < 0 or class_id >= y_pred_proba.shape[1]:
        raise ValueError(f"class_id {class_id} is out of bounds for predictions")
    predicted_classes = np.argmax(y_pred_proba, axis=1)
    predicted_confidences = np.max(y_pred_proba, axis=1)
    true_class_probs = y_pred_proba[np.arange(len(y_true)), y_true_indices]
    is_true_class = y_true_indices == class_id
    is_pred_class = predicted_classes == class_id
    false_negatives = is_true_class & ~is_pred_class
    if np.sum(false_negatives) == 0:
        return pd.DataFrame()
    error_indices = np.where(false_negatives)[0]
    error_magnitudes = (
        predicted_confidences[error_indices] - true_class_probs[error_indices]
    )
    top_error_pos = np.argsort(error_magnitudes)[::-1][:top_n]
    selected_indices = error_indices[top_error_pos]

    df_data = []
    for idx in selected_indices:
        true_class_idx = y_true_indices.iloc[idx]
        pred_class_idx = predicted_classes[idx]
        row = pd.Series(
            {
                "True Class": id2label.get(true_class_idx, true_class_idx),
                "Predicted Class": id2label.get(pred_class_idx, pred_class_idx),
                "Predicted Class Probability": predicted_confidences[idx],
                "True Class Probability": true_class_probs[idx],
                "Error": float(predicted_confidences[idx] - true_class_probs[idx]),
            },
            name=y_true.index[idx],
        )
        df_data.append(row)

    df = pd.DataFrame(df_data)
    return df


def display_classification_errors(
    errors_by_class: dict, id2label: dict[int, str]
) -> None:
    """Display styled tables of classification errors for each class.

    Args:
        errors_by_class (dict): Dictionary mapping class_id to error DataFrame from get_top_classification_errors.
        id2label (dict[int, str]): Mapping from class index to class name.

    Returns:
        None: This function renders styled tables (best viewed in notebooks).
    """

    def _style(df: pd.DataFrame, title: str):
        fmt: dict[str, str] = {}
        for col in df.columns:
            if df[col].dtype.kind in "fc":
                if "Probability" in col:
                    fmt[col] = "{:.4f}"
        if df.empty:
            empty_df = pd.DataFrame({"No errors found": [""]})
            styler = empty_df.style.set_caption(title)
        else:
            styler = df.style.set_caption(title).format(fmt)

        prob_cols = [
            c for c in df.columns if "Probability" in c and df[c].dtype.kind in "fc"
        ]
        if prob_cols:
            styler = styler.background_gradient(
                subset=prob_cols, cmap="RdYlGn", vmin=0, vmax=1
            )

        if "Error" in df.columns:
            styler = styler.bar(
                subset=["Error"], align="mid", color="#d65f5f", vmin=0, vmax=1
            )

        styler = styler.set_properties(
            **{
                "text-align": "left",
                "font-family": "Times New Roman",
                "font-size": "1.25em",
                "background-color": "#f9f9f9",
                "border": "3px solid #ddd",
                "color": "#333",
            }
        ).set_table_styles(
            [
                {
                    "selector": "th",
                    "props": [
                        ("font-weight", "bold"),
                        ("border", "3px solid #ddd"),
                        ("background-color", "#4a90e2"),
                        ("color", "white"),
                        ("text-align", "center"),
                        ("font-size", "1.2em"),
                        ("padding", "8px"),
                    ],
                },
                {
                    "selector": "caption",
                    "props": [
                        ("font-size", "1.4em"),
                        ("font-weight", "bold"),
                        ("color", "#4a90e2"),
                        ("margin-bottom", "12px"),
                    ],
                },
            ]
        )
        return styler

    for class_id, class_name in id2label.items():
        if class_id in errors_by_class:
            df = errors_by_class[class_id]
            title = f"🎯 Class '{class_name}' - Top {len(df)} Classification Errors"
            display(_style(df, title))


if __name__ == "__main__":
    rng = np.random.default_rng(42)
    # Regression example
    y_true_reg = pd.Series(rng.normal(loc=100.0, scale=10.0, size=50))
    y_pred_reg = y_true_reg + rng.normal(loc=0.0, scale=5.0, size=50)
    under_df, over_df = get_top_regression_under_over_errors(
        y_true_reg, y_pred_reg, top_n=5
    )
    display_regression_under_over_errors(under_df, over_df)

    # Classification example (binary)
    y_true_cls = pd.Series(rng.integers(0, 2, size=60))
    # Simulate probabilities with some signal
    raw_scores = y_true_cls + rng.normal(0, 0.8, size=60)
    prob_pos = 1 / (1 + np.exp(-raw_scores))
    errors_by_class = {}
    id2label = {0: "Negative", 1: "Positive"}
    for class_id, class_name in id2label.items():
        df = get_top_classification_errors(
            y_true_cls, prob_pos, class_id=class_id, id2label=id2label, top_n=10
        )
        errors_by_class[class_id] = df
    display_classification_errors(errors_by_class, id2label)
