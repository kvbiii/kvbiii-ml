from __future__ import annotations

import numpy as np
import pandas as pd
try:  # optional interactive dependency
    from IPython.display import display  # type: ignore
except Exception:  # pragma: no cover
    def display(obj):  # type: ignore
        print(obj)


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
        # Build styler
        styler = df.style.set_caption(title).format(fmt)
        numeric_cols = [c for c in df.columns if df[c].dtype.kind in "fc"]
        error_cols = [c for c in df.columns if "Error" in c]
        if error_cols:
            # Larger absolute error -> darker (reverse green-red)
            styler = styler.background_gradient(
                subset=error_cols, cmap="RdYlGn_r"
            )
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
    y_pred_proba: np.ndarray | pd.DataFrame,
    top_n: int = 15,
    positive_class: int | str | None = None,
) -> pd.DataFrame:
    """Compute top misclassified samples for (binary or multi-class) classification.

    For binary classification, probabilities (shape (n_samples, 2) or (n_samples,))
    are used to derive predicted labels. For multi-class, argmax is applied.
    The returned DataFrame is sorted by model confidence in the *wrong* class
    (largest probability assigned to the predicted but incorrect label).

    Args:
        y_true (pd.Series): True class labels.
        y_pred_proba (np.ndarray | pd.DataFrame): Predicted class probabilities.
        top_n (int, optional): Number of misclassified rows to return. Defaults to 15.
        positive_class (int | str | None, optional): Explicit positive class for
            binary one-column probability input. If None and a single probability
            column is provided, assumes positive class is 1.

    Returns:
        pd.DataFrame: Misclassified samples with columns: true, predicted, confidence,
            probability_true_class, probability_predicted_class, margin (prob_pred - prob_true).
    """
    if not isinstance(y_true, pd.Series):
        raise TypeError("y_true must be a pandas Series.")
    proba = (
        y_pred_proba.values
        if isinstance(y_pred_proba, pd.DataFrame)
        else np.asarray(y_pred_proba)
    )
    if proba.ndim == 1:  # single probability (positive class)
        pos = 1 if positive_class is None else positive_class
        neg = 0 if pos != 0 else 1
        proba = np.vstack([1 - proba, proba]).T
        classes = [neg, pos]
    else:
        classes = list(range(proba.shape[1]))
    y_pred = np.array(classes)[np.argmax(proba, axis=1)]
    mis_mask = y_pred != y_true.values
    if not mis_mask.any():
        return pd.DataFrame(
            columns=[
                "True Label",
                "Predicted Label",
                "Confidence (Pred)",
                "Prob True Label",
                "Prob Pred Label",
                "Margin",
            ]
        )
    proba_true = proba[np.arange(len(proba)), y_true.map({c: i for i, c in enumerate(classes)}).values]
    proba_pred = proba[np.arange(len(proba)), np.argmax(proba, axis=1)]
    margin = proba_pred - proba_true
    df = pd.DataFrame(
        {
            "True Label": y_true.values,
            "Predicted Label": y_pred,
            "Confidence (Pred)": proba_pred,
            "Prob True Label": proba_true,
            "Prob Pred Label": proba_pred,
            "Margin": margin,
        }
    )
    df = df.loc[mis_mask]
    df = df.sort_values(by=["Confidence (Pred)", "Margin"], ascending=False).head(top_n)
    return df


class ErrorDiagnostics:
    """Unified error diagnostics helper for regression and classification.

    Provides a minimal API required by tests: initialization with problem_type
    and a compute_errors method producing an error DataFrame.
    """

    def __init__(self, problem_type: str):
        if problem_type not in {"classification", "regression"}:
            raise ValueError("problem_type must be 'classification' or 'regression'")
        self.problem_type = problem_type

    def compute_errors(
        self,
        y_true: np.ndarray | pd.Series,
        y_pred: np.ndarray | pd.Series,
        X: pd.DataFrame | None = None,
        probas: np.ndarray | None = None,
    ) -> pd.DataFrame:
        y_true_s = pd.Series(y_true).reset_index(drop=True)
        y_pred_s = pd.Series(y_pred).reset_index(drop=True)
        df = pd.DataFrame({"y_true": y_true_s, "y_pred": y_pred_s})
        if self.problem_type == "classification":
            df["error"] = (df["y_true"] != df["y_pred"]).astype(int)
            if probas is not None:
                p = np.asarray(probas)
                if p.ndim == 1:
                    # binary positive class prob
                    prob_true = np.where(df["y_true"].values == 1, p, 1 - p)
                else:
                    # assume column order matches sorted unique classes
                    classes = sorted(df["y_true"].unique())
                    class_index = {c: i for i, c in enumerate(classes)}
                    idx = [class_index[c] for c in df["y_true"].values]
                    prob_true = p[np.arange(len(p)), idx]
                df["proba_true_class"] = prob_true
        else:  # regression
            err = y_pred_s - y_true_s
            df["error"] = err
            df["absolute_error"] = err.abs()
            df["squared_error"] = err**2
            with np.errstate(divide="ignore", invalid="ignore"):
                df["percentage_error"] = np.where(
                    y_true_s == 0, np.nan, (err.abs() / y_true_s.abs()) * 100.0
                )
        if X is not None:
            df = pd.concat([df, X.reset_index(drop=True)], axis=1)
        return df

    @staticmethod
    def regression_under_over_errors(y_true, y_pred, top_n=15, log=False):
        return get_top_regression_under_over_errors(y_true, y_pred, top_n=top_n, log=log)

    @staticmethod
    def classification_errors(y_true, y_pred_proba, top_n=15, positive_class=None):
        return get_top_classification_errors(y_true, y_pred_proba, top_n=top_n, positive_class=positive_class)


def display_classification_errors(errors_df: pd.DataFrame) -> None:
    """Display styled table of misclassified samples.

    Args:
        errors_df (pd.DataFrame): Output from ``get_top_classification_errors``.

    Returns:
        None: Renders styled table.
    """
    if errors_df.empty:
        display("No misclassifications found.")
        return
    fmt = {}
    for col in errors_df.columns:
        if errors_df[col].dtype.kind in "fc":
            if "Prob" in col or "Confidence" in col:
                fmt[col] = "{:.4f}"
            elif col == "Margin":
                fmt[col] = "{:+.4f}"
    styler = errors_df.style.set_caption(
        f"🔍 Top {len(errors_df)} Misclassified Samples (highest confidence)"
    ).format(fmt)
    # Emphasize high confidence wrong predictions
    if "Confidence (Pred)" in errors_df.columns:
        styler = styler.background_gradient(
            subset=["Confidence (Pred)"], cmap="OrRd"
        )
    if "Margin" in errors_df.columns:
        styler = styler.bar(subset=["Margin"], align="mid", color=["#d73027", "#1a9850"])
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
    display(styler)


if __name__ == "__main__":
    rng = np.random.default_rng(42)
    # Regression example
    y_true_reg = pd.Series(rng.normal(loc=100.0, scale=10.0, size=50))
    y_pred_reg = y_true_reg + rng.normal(loc=0.0, scale=5.0, size=50)
    under_df, over_df = get_top_regression_under_over_errors(y_true_reg, y_pred_reg, top_n=5)
    display_regression_under_over_errors(under_df, over_df)

    # Classification example (binary)
    y_true_cls = pd.Series(rng.integers(0, 2, size=60))
    # Simulate probabilities with some signal
    raw_scores = y_true_cls + rng.normal(0, 0.8, size=60)
    prob_pos = 1 / (1 + np.exp(-raw_scores))
    mis_cls_df = get_top_classification_errors(y_true_cls, prob_pos, top_n=10)
    display_classification_errors(mis_cls_df)
