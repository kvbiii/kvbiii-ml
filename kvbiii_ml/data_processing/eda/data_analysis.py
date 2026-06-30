from ast import literal_eval

import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency, ks_2samp, kruskal, wasserstein_distance
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression


class DataAnalyzer:
    """A comprehensive data analysis class for exploratory data analysis (EDA).

    This class provides methods for analyzing different types of data features including
    numerical, categorical, and time series data. It generates styled DataFrames with
    statistical summaries and visualizations for better data understanding.
    """

    @staticmethod
    def base_information(data: pd.DataFrame) -> pd.DataFrame:
        """Generates base information about the dataset.

        Args:
            data (pd.DataFrame): The input DataFrame to analyze.

        Returns:
            pd.DataFrame: A styled DataFrame with base information about each feature
                including data types, missing values, unique values, and counts.
        """
        df = pd.DataFrame(data.dtypes, columns=["dtypes"])
        df["Number of missing values"] = data.isna().sum()
        df["Percentage of missing values"] = data.isna().sum() / data.shape[0] * 100
        df["Unique values"] = data.nunique().values
        df["Count"] = data.count().values
        df = df.reset_index().rename(columns={"index": "Feature"})
        styled_df = (
            df.style.set_caption("📝 Base Information")
            .format(
                {
                    "Number of missing values": "{:,.0f}",
                    "Percentage of missing values": "{:.2f}%",
                    "Unique values": "{:,.0f}",
                    "Count": "{:,.0f}",
                }
            )
            .bar(
                subset=["Percentage of missing values"],
                color="#4a90e2",
                vmin=0,
                vmax=100,
            )
            .set_properties(
                **{
                    "text-align": "left",
                    "font-family": "Times New Roman",
                    "font-size": "1.25em",
                    "background-color": "#f9f9f9",
                    "border": "1px solid #ddd",
                    "color": "#333",
                }
            )
            .set_table_styles(
                [
                    {
                        "selector": "th",
                        "props": [
                            ("font-weight", "bold"),
                            ("border", "1px solid #ddd"),
                        ],
                    }
                ]
            )
        )
        return styled_df

    @staticmethod
    def get_categorical_features(
        df: pd.DataFrame, unique_threshold: int = 100
    ) -> list[str]:
        """Gets categorical features from the DataFrame.

        Args:
            df (pd.DataFrame): The input DataFrame.
            unique_threshold (int, optional): Maximum number of unique values for a
                feature to be considered categorical. Defaults to 100.

        Returns:
            list[str]: List of categorical feature names.
        """
        categorical_features = df.select_dtypes(
            include=["object", "string", "category"]
        ).columns
        categorical_features = [
            col for col in categorical_features if df[col].nunique() < unique_threshold
        ]
        categorical_features = list(
            set(categorical_features) | set(df.columns[df.nunique() == 2])
        )
        return sorted(categorical_features)

    @staticmethod
    def extract_unique_items(item_list: str | list) -> list[str]:
        """Extracts unique items from a string or list representation.

        Args:
            item_list (str | list): A string representation of a list or an actual list.

        Returns:
            list[str]: A list of unique items, stripped of whitespace and converted
                to lowercase.
        """
        if pd.isna(item_list) or item_list == "Not Provided":
            return []
        if isinstance(item_list, str):
            try:
                item_list = literal_eval(item_list)
            except (ValueError, SyntaxError):
                return []
        if not isinstance(item_list, list):
            return []
        return list({str(feature).strip().lower() for feature in item_list})

    @staticmethod
    def describe_numerical_feature(data: pd.DataFrame, feature: str) -> pd.DataFrame:
        """Provides enhanced numerical feature analysis with styling.

        Args:
            data (pd.DataFrame): The input DataFrame.
            feature (str): The numerical feature name to analyze.

        Returns:
            pd.DataFrame: A styled DataFrame with comprehensive statistics including
                descriptive statistics, percentiles, variance, skewness, and kurtosis.
        """
        stats = data[feature].describe().T
        stats["5%"] = data[feature].quantile(0.05)
        stats["95%"] = data[feature].quantile(0.95)
        stats["variance"] = data[feature].var()
        stats["skewness"] = data[feature].skew()
        stats["kurtosis"] = data[feature].kurtosis()
        missing_count = data[feature].isna().sum()
        missing_pct = missing_count / len(data) * 100
        stats["Missing"] = missing_count
        stats["Missing (%)"] = missing_pct
        stats = stats.rename(
            {
                "count": "Count",
                "mean": "Mean",
                "std": "Std",
                "min": "Min",
                "25%": "Q1",
                "50%": "Median",
                "75%": "Q3",
                "max": "Max",
            }
        )
        stats = stats[
            [
                "Count",
                "Mean",
                "Std",
                "Min",
                "5%",
                "Q1",
                "Median",
                "Q3",
                "95%",
                "Max",
                "variance",
                "skewness",
                "kurtosis",
                "Missing",
                "Missing (%)",
            ]
        ]
        stats_df = stats.to_frame().T

        styled_stats = (
            stats_df.style.set_caption(f"📈 {feature} Statistics")
            .format(
                {
                    "Count": "{:,.0f}",
                    "Mean": "{:,.2f}",
                    "Std": "{:,.2f}",
                    "Min": "{:,.2f}",
                    "5%": "{:,.2f}",
                    "Q1": "{:,.2f}",
                    "Median": "{:,.2f}",
                    "Q3": "{:,.2f}",
                    "95%": "{:,.2f}",
                    "Max": "{:,.2f}",
                    "variance": "{:,.2f}",
                    "skewness": "{:,.2f}",
                    "kurtosis": "{:,.2f}",
                    "Missing": "{:,.0f}",
                    "Missing (%)": "{:.2f}%",
                }
            )
            .set_properties(
                **{
                    "text-align": "left",
                    "font-family": "Times New Roman",
                    "font-size": "1.25em",
                    "background-color": "#f9f9f9",
                    "border": "1px solid #ddd",
                    "color": "#333",
                }
            )
            .set_table_styles(
                [
                    {
                        "selector": "th",
                        "props": [
                            ("font-weight", "bold"),
                            ("border", "1px solid #ddd"),
                        ],
                    }
                ]
            )
        )
        return styled_stats

    @staticmethod
    def describe_categorical_feature(
        data: pd.DataFrame, feature: str, top_n: int = 10, show_null: bool = True
    ) -> pd.DataFrame:
        """Provides enhanced categorical feature analysis with styling.

        Args:
            data (pd.DataFrame): The input DataFrame.
            feature (str): The categorical feature name to analyze.
            top_n (int, optional): Number of top categories to display. Defaults to 10.\
            show_null (bool, optional): Whether to include null values in the analysis.

        Returns:
            pd.DataFrame: A styled DataFrame with category distribution including
                counts and percentages.
        """
        data_copy = data.copy()
        counts = data_copy[feature].value_counts(dropna=not show_null)
        total = len(data_copy)
        if len(counts) > top_n:
            top_counts = counts[:top_n]
            other_count = counts[top_n:].sum()
            counts = pd.concat(
                [top_counts, pd.Series({f"Other ({len(counts) - top_n})": other_count})]
            )

        percentages = (counts / total * 100).round(2)

        stats = pd.DataFrame(
            {
                "Category": counts.index.tolist(),
                "Count": counts.values.tolist(),
                "Percentage (%)": percentages.values.tolist(),
            }
        )

        styled_stats = (
            stats.style.set_caption(f"📊 {feature} Distribution")
            .format({"Count": "{:,.0f}", "Percentage (%)": "{:.2f}%"})
            .bar(subset=["Percentage (%)"], color="#4a90e2", vmin=0, vmax=100)
            .set_properties(
                **{
                    "text-align": "left",
                    "font-family": "Times New Roman",
                    "font-size": "1.25em",
                    "background-color": "#f9f9f9",
                    "border": "1px solid #ddd",
                    "color": "#333",
                }
            )
            .set_table_styles(
                [
                    {
                        "selector": "th",
                        "props": [
                            ("font-weight", "bold"),
                            ("border", "1px solid #ddd"),
                        ],
                    }
                ]
            )
        )
        return styled_stats

    @staticmethod
    def describe_time_series_feature(
        data: pd.DataFrame, feature: str, agg_freq: str = "ME"
    ) -> pd.DataFrame:
        """Provides enhanced time series feature analysis with styling.

        Args:
            data (pd.DataFrame): The input DataFrame.
            feature (str): The time series feature name to analyze.
            agg_freq (str, optional): Aggregation frequency ('ME', 'YE', 'D', 'W', 'H').
                Defaults to "ME".

        Returns:
            pd.DataFrame: A styled DataFrame with temporal distribution analysis.

        Raises:
            ValueError: If agg_freq is not one of the supported values.
        """
        data_copy = data.copy().reset_index(drop=True)
        missing_count = data_copy[feature].isna().sum()

        if agg_freq == "ME":
            counts = data_copy[feature].dt.month.value_counts().sort_index()
            label, caption = "Month", "📅 {} Monthly Distribution"
        elif agg_freq == "YE":
            counts = data_copy[feature].dt.year.value_counts().sort_index()
            label, caption = "Year", "📅 {} Yearly Distribution"
        elif agg_freq == "D":
            counts = data_copy[feature].dt.day.value_counts().sort_index()
            label, caption = "Day", "📅 {} Daily Distribution"
        elif agg_freq == "W":
            counts = (
                data_copy[feature].dt.isocalendar().week.value_counts().sort_index()
            )
            label, caption = "Week", "📅 {} Weekly Distribution"
        elif agg_freq == "H":
            counts = data_copy[feature].dt.hour.value_counts().sort_index()
            label, caption = "Hour", "🕒 {} Hourly Distribution"
        else:
            raise ValueError(
                "Invalid aggregation frequency. Use 'ME', 'YE', 'D', 'W', or 'H'."
            )

        df = pd.DataFrame({label: counts.index.astype(int), "Count": counts.values})
        if missing_count > 0:
            df = pd.concat(
                [df, pd.DataFrame([{label: "Unknown", "Count": missing_count}])],
                ignore_index=True,
            )

        styled_df = (
            df.style.set_caption(caption.format(feature))
            .format({"Count": "{:,.0f}"})
            .bar(subset=["Count"], color="#4a90e2", vmin=0, vmax=df["Count"].max())
            .set_properties(
                **{
                    "text-align": "left",
                    "font-family": "Times New Roman",
                    "font-size": "1.25em",
                    "background-color": "#f9f9f9",
                    "border": "1px solid #ddd",
                    "color": "#333",
                }
            )
            .set_table_styles(
                [
                    {
                        "selector": "th",
                        "props": [
                            ("font-weight", "bold"),
                            ("border", "1px solid #ddd"),
                        ],
                    }
                ]
            )
        )
        return styled_df

    @staticmethod
    def _skewness_cell_color(val: float) -> str:
        """Maps an absolute skewness value to a background colour string."""
        abs_val = abs(val)
        if abs_val < 0.5:
            r, g, b = 76, 175, 80
        elif abs_val < 1.0:
            t = (abs_val - 0.5) / 0.5
            r = int(76 + t * (255 - 76))
            g = int(175 + t * (193 - 175))
            b = int(80 + t * (7 - 80))
        elif abs_val < 5.0:
            t = min((abs_val - 1.0) / 4.0, 1.0)
            r = 255
            g = int(193 - t * 193)
            b = 7
        else:
            r, g, b = 198, 40, 40
        return f"background-color: rgb({r},{g},{b}); color: white; font-weight: bold; text-align: center"

    @staticmethod
    def numerical_feature_statistics(
        data: pd.DataFrame,
        numerical_features: list[str],
        skew_threshold: float = 1.0,
        kurtosis_threshold: float = 3.0,
        zero_pct_threshold: float = 5.0,
    ) -> pd.DataFrame:
        """Generates a summary statistics table for all numerical features.

        Computes mean, median, std, min, max, skewness, kurtosis, zero percentage,
        IQR-based outlier count and percentage, and a plain-text flag column that
        highlights distribution issues.  The returned Styler uses the same
        Times-New-Roman / navy-header theme as ``base_information`` but at a larger
        font size, with alternating row shading, hover highlighting, and a
        colour-coded skewness column (green → orange → red by absolute magnitude).

        Args:
            data (pd.DataFrame): The input DataFrame to analyse.
            numerical_features (list[str]): List of numerical feature names to analyze.
            skew_threshold (float, optional): Absolute skewness above which the
                ``HIGH SKEW`` flag is raised. Defaults to 1.0.
            kurtosis_threshold (float, optional): Excess kurtosis above which the
                ``HEAVY TAILS`` flag is raised. Defaults to 3.0.
            zero_pct_threshold (float, optional): Percentage of zeros above which the
                ``ZERO-INFLATED`` flag is raised. Defaults to 5.0.

        Returns:
            pd.DataFrame: A styled DataFrame with one row per numerical feature.
        """
        rows = []
        for col in numerical_features:
            series = data[col].dropna()
            q1 = series.quantile(0.25)
            q3 = series.quantile(0.75)
            iqr = q3 - q1
            lower_fence = q1 - 1.5 * iqr
            upper_fence = q3 + 1.5 * iqr
            n_outliers = int(((series < lower_fence) | (series > upper_fence)).sum())
            pct_outliers = n_outliers / max(len(data), 1) * 100
            pct_zeros = (series == 0).sum() / max(len(data), 1) * 100
            skewness = series.skew()
            kurtosis = series.kurtosis()

            flags = []
            if skewness >= skew_threshold:
                flags.append("HIGH RIGHT SKEW")
            elif skewness <= -skew_threshold:
                flags.append("HIGH LEFT SKEW")
            if pct_zeros >= zero_pct_threshold:
                flags.append("ZERO-INFLATED")
            if kurtosis >= kurtosis_threshold:
                flags.append("HEAVY TAILS")

            rows.append(
                {
                    "Feature": col,
                    "Min": series.min(),
                    "Mean": series.mean(),
                    "Median": series.median(),
                    "Max": series.max(),
                    "Std": series.std(),
                    "Skewness": skewness,
                    "Kurtosis": kurtosis,
                    "Outliers IQR (%)": f"{n_outliers} ({pct_outliers:.2f}%)",
                    "Flags": "; ".join(flags),
                }
            )

        df = pd.DataFrame(rows)

        table_styles = [
            {
                "selector": "caption",
                "props": [
                    ("font-family", "Times New Roman"),
                    ("font-size", "1.7em"),
                    ("font-weight", "bold"),
                    ("color", "#1e3a5f"),
                    ("padding", "14px 0 10px 4px"),
                    ("text-align", "left"),
                    ("letter-spacing", "0.04em"),
                ],
            },
            {
                "selector": "thead th",
                "props": [
                    ("background-color", "#1e3a5f"),
                    ("color", "white"),
                    ("font-family", "Times New Roman"),
                    ("font-size", "1.3em"),
                    ("font-weight", "bold"),
                    ("padding", "11px 16px"),
                    ("border", "none"),
                    ("text-align", "center"),
                    ("letter-spacing", "0.04em"),
                    ("white-space", "nowrap"),
                ],
            },
            {
                "selector": "tbody tr:nth-child(even)",
                "props": [("background-color", "#eef3fb")],
            },
            {
                "selector": "tbody tr:nth-child(odd)",
                "props": [("background-color", "#ffffff")],
            },
            {
                "selector": "tbody tr:hover",
                "props": [("background-color", "#d0e4f7")],
            },
            {
                "selector": "",
                "props": [
                    ("border-collapse", "collapse"),
                    ("box-shadow", "0 4px 14px rgba(0,0,0,0.10)"),
                    ("width", "100%"),
                ],
            },
        ]

        styled = (
            df.style.set_caption("📊 Numerical Feature Statistics")
            .format(
                {
                    "Min": "{:.3f}",
                    "Mean": "{:.3f}",
                    "Median": "{:.3f}",
                    "Max": "{:.3f}",
                    "Std": "{:.3f}",
                    "Skewness": "{:.3f}",
                    "Kurtosis": "{:.3f}",
                }
            )
            .map(DataAnalyzer._skewness_cell_color, subset=["Skewness"])
            .set_table_styles(table_styles)
            .set_properties(
                **{
                    "font-family": "Times New Roman",
                    "font-size": "1.4em",
                    "padding": "9px 16px",
                    "border": "none",
                    "border-bottom": "1px solid #dde6f0",
                    "text-align": "center",
                    "color": "#2c3e50",
                }
            )
            .set_properties(
                subset=["Feature"], **{"text-align": "left", "font-weight": "bold"}
            )
            .set_properties(
                subset=["Flags"],
                **{
                    "text-align": "left",
                    "font-style": "italic",
                    "color": "#7b3f00",
                    "white-space": "nowrap",
                },
            )
        )
        return styled

    @staticmethod
    def _effect_size_cell_color(val: float) -> str:
        """Maps an eta / epsilon squared value to a background colour string.

        Thresholds follow Cohen (1988): negligible < 0.01, small < 0.06,
        medium < 0.14, large ≥ 0.14.
        """
        if val < 0.01:
            r, g, b = 189, 189, 189
        elif val < 0.06:
            r, g, b = 255, 183, 77
        elif val < 0.14:
            r, g, b = 230, 81, 0
        else:
            r, g, b = 56, 142, 60
        return f"background-color: rgb({r},{g},{b}); color: white; font-weight: bold; text-align: center"

    @staticmethod
    def _pvalue_cell_color(val: float) -> str:
        """Maps a p-value to a background colour (green = significant, red = not)."""
        if val < 0.001:
            r, g, b = 27, 94, 32
        elif val < 0.01:
            r, g, b = 56, 142, 60
        elif val < 0.05:
            r, g, b = 104, 159, 56
        elif val < 0.10:
            r, g, b = 245, 127, 23
        else:
            r, g, b = 198, 40, 40
        return f"background-color: rgb({r},{g},{b}); color: white; font-weight: bold; text-align: center"

    @staticmethod
    def _bin_regression_target(
        data: pd.DataFrame, target: str, n_quantiles: int
    ) -> tuple[pd.DataFrame, list[str]]:
        """Bins a continuous regression target into exactly n_quantiles equal-frequency bands.

        Ranks the target with ``method="first"`` to break all ties, then applies
        ``pd.qcut`` on the ranks.  This guarantees exactly ``n_quantiles`` bins even
        when the target contains many duplicate values.  A ``__target_bin__`` column
        (string labels of the form ``"Q{i} [min, max]"``) is added to a copy of
        ``data``.  Rows where ``target`` is NaN receive NaN bin assignments.

        Args:
            data (pd.DataFrame): The input DataFrame.
            target (str): Continuous target column to bin.
            n_quantiles (int): Number of equal-frequency quantile bands.

        Returns:
            tuple[pd.DataFrame, list[str]]: A copy of ``data`` with ``__target_bin__``
                added, and the ordered list of quantile label strings.
        """
        data_copy = data.copy()
        data_copy["__bin_idx__"] = pd.qcut(
            data_copy[target].rank(method="first"), q=n_quantiles, labels=False
        )
        quantile_labels = []
        for bin_idx in range(n_quantiles):
            bin_mask = data_copy["__bin_idx__"] == bin_idx
            vals = data_copy.loc[bin_mask, target]
            quantile_labels.append(
                f"Q{bin_idx + 1} [{vals.min():.3f}, {vals.max():.3f}]"
            )
        data_copy["__target_bin__"] = data_copy["__bin_idx__"].map(
            dict(enumerate(quantile_labels))
        )
        return data_copy.drop(columns=["__bin_idx__"]), quantile_labels

    @staticmethod
    def describe_feature_by_target(
        data: pd.DataFrame,
        feature: str,
        target: str,
        problem_type: str,
        n_quantiles: int = 4,
    ) -> pd.DataFrame:
        """Per-group descriptive statistics for a single numerical feature.

        For classification, groups by each unique target class.  For regression,
        bins the continuous target into exactly ``n_quantiles`` equal-frequency bands by
        first ranking the target values (``rank(method="first")`` breaks all ties by
        observation order) and then applying ``pd.qcut`` on the ranks.  Because ranks are
        always unique, this guarantees exactly ``n_quantiles`` bins even when the target
        contains many duplicate values (e.g. zero-inflated distributions).  Each bin label
        shows the actual min/max target value within that band.

        An *Overall* summary row is appended at the bottom.  The skewness column is
        colour-coded (green → orange → red) and the Overall row is highlighted with a
        distinct light-blue background and top border for easy visual separation.

        Args:
            data (pd.DataFrame): The input DataFrame.
            feature (str): Numerical feature to analyse.
            target (str): Target column name.
            problem_type (str): Either ``"classification"`` or ``"regression"``.
            n_quantiles (int, optional): Number of equal-frequency quantile bands used
                when ``problem_type="regression"``. Defaults to 4.

        Returns:
            pd.DataFrame: A styled DataFrame with one row per group plus Overall.

        Raises:
            ValueError: If ``problem_type`` is not ``"classification"`` or
                ``"regression"``.
        """
        if problem_type not in {"classification", "regression"}:
            raise ValueError(
                "problem_type must be either 'classification' or 'regression'."
            )

        rows: list[dict] = []

        if problem_type == "classification":
            target_col_name = "Target"
            caption = f"🎯 {feature} — Per-Class Statistics"
            for cls in sorted(data[target].dropna().unique()):
                mask = data[target] == cls
                series = data.loc[mask, feature].dropna()
                rows.append(
                    {
                        target_col_name: cls,
                        "Count": len(series),
                        "Mean": series.mean(),
                        "Std": series.std(),
                        "Min": series.min(),
                        "Q25": series.quantile(0.25),
                        "Median": series.median(),
                        "Q75": series.quantile(0.75),
                        "Max": series.max(),
                        "Skewness": series.skew(),
                        "Missing (%)": data.loc[mask, feature].isna().mean() * 100,
                    }
                )
        else:
            target_col_name = "Target Quantile"
            caption = f"🎯 {feature} — Per-Quantile Statistics (Target: {target})"
            data_copy, quantile_labels = DataAnalyzer._bin_regression_target(
                data, target, n_quantiles
            )
            for bin_label in quantile_labels:
                mask = data_copy["__target_bin__"] == bin_label
                series = data_copy.loc[mask, feature].dropna()
                rows.append(
                    {
                        target_col_name: bin_label,
                        "Count": len(series),
                        "Mean": series.mean(),
                        "Std": series.std(),
                        "Min": series.min(),
                        "Q25": series.quantile(0.25),
                        "Median": series.median(),
                        "Q75": series.quantile(0.75),
                        "Max": series.max(),
                        "Skewness": series.skew(),
                        "Missing (%)": data_copy.loc[mask, feature].isna().mean() * 100,
                    }
                )

        overall = data[feature].dropna()
        rows.append(
            {
                target_col_name: "Overall",
                "Count": len(overall),
                "Mean": overall.mean(),
                "Std": overall.std(),
                "Min": overall.min(),
                "Q25": overall.quantile(0.25),
                "Median": overall.median(),
                "Q75": overall.quantile(0.75),
                "Max": overall.max(),
                "Skewness": overall.skew(),
                "Missing (%)": data[feature].isna().mean() * 100,
            }
        )

        df = pd.DataFrame(rows)

        def _highlight_overall(row: pd.Series) -> list[str]:
            """Highlights the Overall summary row."""
            if str(row[target_col_name]) != "Overall":
                return [""] * len(row)
            return [
                "background-color: #d4e6f1; font-weight: bold; border-top: 2px solid #1e3a5f"
            ] * len(row)

        table_styles = [
            {
                "selector": "caption",
                "props": [
                    ("font-family", "Times New Roman"),
                    ("font-size", "1.7em"),
                    ("font-weight", "bold"),
                    ("color", "#1e3a5f"),
                    ("padding", "14px 0 10px 4px"),
                    ("text-align", "left"),
                    ("letter-spacing", "0.04em"),
                ],
            },
            {
                "selector": "thead th",
                "props": [
                    ("background-color", "#1e3a5f"),
                    ("color", "white"),
                    ("font-family", "Times New Roman"),
                    ("font-size", "1.3em"),
                    ("font-weight", "bold"),
                    ("padding", "11px 16px"),
                    ("border", "none"),
                    ("text-align", "center"),
                    ("letter-spacing", "0.04em"),
                    ("white-space", "nowrap"),
                ],
            },
            {
                "selector": "tbody tr:nth-child(even)",
                "props": [("background-color", "#eef3fb")],
            },
            {
                "selector": "tbody tr:nth-child(odd)",
                "props": [("background-color", "#ffffff")],
            },
            {
                "selector": "tbody tr:hover",
                "props": [("background-color", "#d0e4f7")],
            },
            {
                "selector": "",
                "props": [
                    ("border-collapse", "collapse"),
                    ("box-shadow", "0 4px 14px rgba(0,0,0,0.10)"),
                    ("width", "100%"),
                ],
            },
        ]

        styled = (
            df.style.set_caption(caption)
            .format(
                {
                    "Count": "{:,.0f}",
                    "Mean": "{:.3f}",
                    "Std": "{:.3f}",
                    "Min": "{:.3f}",
                    "Q25": "{:.3f}",
                    "Median": "{:.3f}",
                    "Q75": "{:.3f}",
                    "Max": "{:.3f}",
                    "Skewness": "{:.3f}",
                    "Missing (%)": "{:.2f}%",
                }
            )
            .map(DataAnalyzer._skewness_cell_color, subset=["Skewness"])
            .apply(_highlight_overall, axis=1)
            .set_table_styles(table_styles)
            .set_properties(
                **{
                    "font-family": "Times New Roman",
                    "font-size": "1.4em",
                    "padding": "9px 16px",
                    "border": "none",
                    "border-bottom": "1px solid #dde6f0",
                    "text-align": "center",
                    "color": "#2c3e50",
                }
            )
            .set_properties(
                subset=[target_col_name],
                **{"text-align": "left", "font-weight": "bold"},
            )
        )
        return styled

    @staticmethod
    def numerical_target_association(
        data: pd.DataFrame,
        numerical_features: list[str],
        target: str,
        problem_type: str,
        n_quantiles: int = 4,
    ) -> pd.DataFrame:
        """Cross-group comparison of numerical features against a target.

        For every numerical column computes per-group median and IQR alongside three
        complementary effect-size metrics:

        - **Eta squared (η²)**: proportion of total variance explained by group
          membership, derived from one-way ANOVA sum-of-squares.  Thresholds:
          small < 0.06, medium < 0.14, large ≥ 0.14.
        - **Epsilon squared (ε²)**: non-parametric effect size from the Kruskal-Wallis
          H statistic, ``(H − k + 1) / (n − k)``.  More robust to outliers and skewed
          distributions than η².
        - **Median range / mean IQR**: ``(max_group_median − min_group_median) /
          mean_group_IQR``.  Answers "how many within-group spreads separate the most
          distant group medians?", making it interpretable regardless of feature scale.

        For ``problem_type="regression"`` the continuous target is first binned into
        ``n_quantiles`` equal-frequency bands via ``_bin_regression_target``; the bands
        are then treated as groups.

        Args:
            data (pd.DataFrame): The input DataFrame.
            numerical_features (list[str]): List of numerical feature names to analyze.
            target (str): Target column name.
            problem_type (str): Either ``"classification"`` or ``"regression"``.
            n_quantiles (int, optional): Number of quantile bands used when
                ``problem_type="regression"``. Defaults to 4.

        Returns:
            pd.DataFrame: A styled DataFrame sorted by η² descending.

        Raises:
            ValueError: If ``problem_type`` is invalid or ``target`` is absent.
        """
        if problem_type not in {"classification", "regression"}:
            raise ValueError(
                "problem_type must be either 'classification' or 'regression'."
            )
        if target not in data.columns:
            raise ValueError(f"Target column '{target}' not found in DataFrame.")

        if problem_type == "regression":
            data_work, classes = DataAnalyzer._bin_regression_target(
                data, target, n_quantiles
            )
            group_col = "__target_bin__"
            caption = (
                f"🔗 Numerical Features — Association with {target} (Quantile Bins)"
            )
        else:
            data_work = data
            classes = sorted(data[target].dropna().unique())
            group_col = target
            caption = f"🔗 Numerical Features — Association with {target}"

        rows = []
        for col in numerical_features:
            clean = data_work[[col, group_col]].dropna()
            groups = [clean.loc[clean[group_col] == cls, col].values for cls in classes]

            row: dict = {"Feature": col}
            for cls, grp in zip(classes, groups):
                row[f"{cls} Median"] = float(np.median(grp)) if len(grp) else np.nan
                row[f"{cls} IQR"] = (
                    float(np.percentile(grp, 75) - np.percentile(grp, 25))
                    if len(grp)
                    else np.nan
                )

            grand_mean = float(clean[col].mean())
            ss_between = sum(
                len(grp) * (float(np.mean(grp)) - grand_mean) ** 2 for grp in groups
            )
            ss_total = float(((clean[col] - grand_mean) ** 2).sum())
            eta_sq = ss_between / ss_total if ss_total > 0 else 0.0

            valid_groups = [g for g in groups if len(g) >= 1]
            if len(valid_groups) >= 2:
                h_stat, p_val = kruskal(*valid_groups)
            else:
                h_stat, p_val = np.nan, np.nan
            n_total = len(clean)
            k = len(valid_groups)
            epsilon_sq = (
                max((h_stat - k + 1) / (n_total - k), 0.0) if n_total > k else 0.0
            )

            cls_medians = [row[f"{cls} Median"] for cls in classes]
            cls_iqrs = [row[f"{cls} IQR"] for cls in classes]
            valid_medians = [m for m in cls_medians if not np.isnan(m)]
            valid_iqrs = [q for q in cls_iqrs if not np.isnan(q)]
            mean_iqr = float(np.mean(valid_iqrs)) if valid_iqrs else 1.0
            median_range_iqr = (
                (max(valid_medians) - min(valid_medians)) / mean_iqr
                if mean_iqr > 0 and len(valid_medians) >= 2
                else 0.0
            )

            row.update(
                {
                    "η²": eta_sq,
                    "ε²": epsilon_sq,
                    "KW p-value": p_val,
                    "Median Range / IQR": median_range_iqr,
                }
            )
            rows.append(row)

        df = (
            pd.DataFrame(rows).sort_values("η²", ascending=False).reset_index(drop=True)
        )

        format_dict: dict[str, str] = {
            "η²": "{:.5f}",
            "ε²": "{:.5f}",
            "KW p-value": "{:.3e}",
            "Median Range / IQR": "{:.5f}",
        }
        for cls in classes:
            format_dict[f"{cls} Median"] = "{:.5f}"
            format_dict[f"{cls} IQR"] = "{:.5f}"

        table_styles = [
            {
                "selector": "caption",
                "props": [
                    ("font-family", "Times New Roman"),
                    ("font-size", "1.7em"),
                    ("font-weight", "bold"),
                    ("color", "#1e3a5f"),
                    ("padding", "14px 0 10px 4px"),
                    ("text-align", "left"),
                    ("letter-spacing", "0.04em"),
                ],
            },
            {
                "selector": "thead th",
                "props": [
                    ("background-color", "#1e3a5f"),
                    ("color", "white"),
                    ("font-family", "Times New Roman"),
                    ("font-size", "1.3em"),
                    ("font-weight", "bold"),
                    ("padding", "11px 16px"),
                    ("border", "none"),
                    ("text-align", "center"),
                    ("letter-spacing", "0.04em"),
                    ("white-space", "nowrap"),
                ],
            },
            {
                "selector": "tbody tr:nth-child(even)",
                "props": [("background-color", "#eef3fb")],
            },
            {
                "selector": "tbody tr:nth-child(odd)",
                "props": [("background-color", "#ffffff")],
            },
            {
                "selector": "tbody tr:hover",
                "props": [("background-color", "#d0e4f7")],
            },
            {
                "selector": "",
                "props": [
                    ("border-collapse", "collapse"),
                    ("box-shadow", "0 4px 14px rgba(0,0,0,0.10)"),
                    ("width", "100%"),
                ],
            },
        ]

        styled = (
            df.style.set_caption(caption)
            .format(format_dict)
            .map(DataAnalyzer._effect_size_cell_color, subset=["η²", "ε²"])
            .map(DataAnalyzer._pvalue_cell_color, subset=["KW p-value"])
            .set_table_styles(table_styles)
            .set_properties(
                **{
                    "font-family": "Times New Roman",
                    "font-size": "1.4em",
                    "padding": "9px 16px",
                    "border": "none",
                    "border-bottom": "1px solid #dde6f0",
                    "text-align": "center",
                    "color": "#2c3e50",
                }
            )
            .set_properties(
                subset=["Feature"], **{"text-align": "left", "font-weight": "bold"}
            )
        )
        return styled

    @staticmethod
    def _association_cell_color(val: float) -> str:
        """Maps Cramér's V or Theil's U to a background colour.

        Thresholds: negligible < 0.05, weak < 0.15, moderate < 0.30, strong ≥ 0.30.
        """
        if val < 0.05:
            r, g, b = 189, 189, 189
        elif val < 0.15:
            r, g, b = 255, 183, 77
        elif val < 0.30:
            r, g, b = 230, 81, 0
        else:
            r, g, b = 56, 142, 60
        return f"background-color: rgb({r},{g},{b}); color: white; font-weight: bold; text-align: center"

    @staticmethod
    def _entropy_cell_color(val: float) -> str:
        """Maps normalised Shannon entropy (0–1) to a colour.

        Green (≥ 0.80, well-distributed) → orange (≥ 0.50) → red (< 0.50, highly concentrated).
        """
        if val >= 0.80:
            r, g, b = 56, 142, 60
        elif val >= 0.50:
            r, g, b = 230, 81, 0
        else:
            r, g, b = 198, 40, 40
        return f"background-color: rgb({r},{g},{b}); color: white; font-weight: bold; text-align: center"

    @staticmethod
    def _compute_theil_u(feature: pd.Series, target: pd.Series) -> float:
        """Uncertainty coefficient U(target|feature): fraction of target entropy explained by feature."""
        contingency = pd.crosstab(feature, target).values.astype(float)
        n = contingency.sum()
        if n == 0:
            return 0.0
        target_marginal = contingency.sum(axis=0) / n
        h_target = -float(
            np.sum(
                np.where(
                    target_marginal > 0, target_marginal * np.log(target_marginal), 0.0
                )
            )
        )
        if h_target == 0.0:
            return 0.0
        row_sums = contingency.sum(axis=1, keepdims=True)
        safe_sums = np.where(row_sums > 0, row_sums, 1.0)
        cond_probs = contingency / safe_sums
        safe_log = np.where(cond_probs > 0, np.log(np.maximum(cond_probs, 1e-300)), 0.0)
        h_rows = -np.sum(cond_probs * safe_log, axis=1)
        weights = contingency.sum(axis=1) / n
        return float((h_target - float(np.dot(weights, h_rows))) / h_target)

    @staticmethod
    def _compute_cramers_v_chi2(
        feature: pd.Series, target: pd.Series
    ) -> tuple[float, float, float]:
        """Returns (Cramér's V, chi-squared statistic, p-value) for two categorical series."""
        contingency = pd.crosstab(feature, target)
        chi2, p_val, _, _ = chi2_contingency(contingency)
        n = contingency.values.sum()
        r, c = contingency.shape
        min_dim = min(r - 1, c - 1)
        v = float(np.sqrt(chi2 / (n * min_dim))) if min_dim > 0 and n > 0 else 0.0
        return v, float(chi2), float(p_val)

    @staticmethod
    def categorical_feature_statistics(
        data: pd.DataFrame,
        categorical_features: list[str],
        rare_threshold: float = 5.0,
    ) -> pd.DataFrame:
        """Generates a summary statistics table for all categorical features.

        For each categorical column (dtype object, category, or bool) computes:
        the number of unique values, mode and its percentage, cumulative top-3
        coverage, normalised Shannon entropy (0 = one dominant value, 1 = perfectly
        uniform), count of rare categories (frequency below ``rare_threshold``%), and
        percentage of missing values.  Rows are sorted by cardinality ascending so
        low-cardinality features appear first.

        Entropy colouring: green (≥ 0.80, diverse) → orange (≥ 0.50) → red (< 0.50,
        highly concentrated / near-constant).

        Args:
            data (pd.DataFrame): The input DataFrame to analyse.
            categorical_features (list[str]): List of categorical feature names to analyze.
            rare_threshold (float, optional): A category is labelled "rare" when its
                frequency is below this percentage of total rows. Defaults to 5.0.

        Returns:
            pd.DataFrame: A styled DataFrame with one row per categorical feature.
        """
        rows = []
        for col in categorical_features:
            series = data[col].dropna()
            n_total = len(data)
            value_counts = series.value_counts()
            n_unique = len(value_counts)

            mode = str(value_counts.index[0])[:25] if n_unique > 0 else "—"
            mode_pct = value_counts.iloc[0] / n_total * 100 if n_unique > 0 else 0.0
            top3_pct = value_counts.head(3).sum() / n_total * 100

            probs = value_counts.values / value_counts.values.sum()
            nonzero_probs = probs[probs > 0]
            h = -float(np.sum(nonzero_probs * np.log(nonzero_probs)))
            h_max = np.log(n_unique) if n_unique > 1 else 1.0
            entropy_norm = h / h_max if h_max > 0 else 0.0

            n_rare = int((value_counts / n_total * 100 < rare_threshold).sum())
            pct_missing = data[col].isna().mean() * 100

            rows.append(
                {
                    "Feature": col,
                    "N Unique": n_unique,
                    "Mode": mode,
                    "Mode %": mode_pct,
                    "Top-3 Coverage %": top3_pct,
                    "Entropy (norm.)": entropy_norm,
                    "Rare Categories": n_rare,
                    "Missing %": pct_missing,
                }
            )

        df = pd.DataFrame(rows).sort_values("N Unique").reset_index(drop=True)

        table_styles = [
            {
                "selector": "caption",
                "props": [
                    ("font-family", "Times New Roman"),
                    ("font-size", "1.7em"),
                    ("font-weight", "bold"),
                    ("color", "#1e3a5f"),
                    ("padding", "14px 0 10px 4px"),
                    ("text-align", "left"),
                    ("letter-spacing", "0.04em"),
                ],
            },
            {
                "selector": "thead th",
                "props": [
                    ("background-color", "#1e3a5f"),
                    ("color", "white"),
                    ("font-family", "Times New Roman"),
                    ("font-size", "1.3em"),
                    ("font-weight", "bold"),
                    ("padding", "11px 16px"),
                    ("border", "none"),
                    ("text-align", "center"),
                    ("letter-spacing", "0.04em"),
                    ("white-space", "nowrap"),
                ],
            },
            {
                "selector": "tbody tr:nth-child(even)",
                "props": [("background-color", "#eef3fb")],
            },
            {
                "selector": "tbody tr:nth-child(odd)",
                "props": [("background-color", "#ffffff")],
            },
            {
                "selector": "tbody tr:hover",
                "props": [("background-color", "#d0e4f7")],
            },
            {
                "selector": "",
                "props": [
                    ("border-collapse", "collapse"),
                    ("box-shadow", "0 4px 14px rgba(0,0,0,0.10)"),
                    ("width", "100%"),
                ],
            },
        ]

        styled = (
            df.style.set_caption("📋 Categorical Feature Statistics")
            .format(
                {
                    "N Unique": "{:,.0f}",
                    "Mode %": "{:.2f}%",
                    "Top-3 Coverage %": "{:.2f}%",
                    "Entropy (norm.)": "{:.3f}",
                    "Rare Categories": "{:,.0f}",
                    "Missing %": "{:.2f}%",
                }
            )
            .map(DataAnalyzer._entropy_cell_color, subset=["Entropy (norm.)"])
            .bar(subset=["Missing %"], color="#4a90e2", vmin=0, vmax=100)
            .set_table_styles(table_styles)
            .set_properties(
                **{
                    "font-family": "Times New Roman",
                    "font-size": "1.4em",
                    "padding": "9px 16px",
                    "border": "none",
                    "border-bottom": "1px solid #dde6f0",
                    "text-align": "center",
                    "color": "#2c3e50",
                }
            )
            .set_properties(subset=["Feature", "Mode"], **{"text-align": "left"})
            .set_properties(subset=["Feature"], **{"font-weight": "bold"})
        )
        return styled

    @staticmethod
    def describe_categorical_by_target(
        data: pd.DataFrame,
        feature: str,
        target: str,
        problem_type: str,
        n_quantiles: int = 4,
        top_n: int = 15,
    ) -> pd.DataFrame:
        """Cross-distribution of a categorical feature against a target.

        Each row corresponds to a unique category of ``feature`` (top ``top_n`` by
        frequency; the remainder are collapsed into an *Other* bucket).  Columns show
        total count and percentage, then for every group the raw count and the *row
        percentage* — i.e. what share of that category's rows fall into each group.

        For ``problem_type="regression"`` the continuous target is first binned into
        ``n_quantiles`` equal-frequency quantile bands via ``_bin_regression_target``;
        the bands become the groups.

        A *Total* row at the bottom shows marginal group distributions.  The per-group
        row-percentage columns are shaded with a blue gradient (darker = higher share,
        normalised within each group column) so over- and under-represented categories
        stand out immediately.

        Args:
            data (pd.DataFrame): The input DataFrame.
            feature (str): Categorical feature to analyse.
            target (str): Target column name.
            problem_type (str): Either ``"classification"`` or ``"regression"``.
            n_quantiles (int, optional): Number of quantile bands used when
                ``problem_type="regression"``. Defaults to 4.
            top_n (int, optional): Max categories before collapsing into "Other".
                Defaults to 15.

        Returns:
            pd.DataFrame: A styled cross-distribution table.

        Raises:
            ValueError: If ``problem_type`` is invalid.
        """
        if problem_type not in {"classification", "regression"}:
            raise ValueError(
                "problem_type must be either 'classification' or 'regression'."
            )

        if problem_type == "regression":
            data_work, classes = DataAnalyzer._bin_regression_target(
                data, target, n_quantiles
            )
            group_col = "__target_bin__"
            caption = f"🔀 {feature} × {target} — Quantile Cross-Distribution"
        else:
            data_work = data
            classes = sorted(data[target].dropna().unique())
            group_col = target
            caption = f"🔀 {feature} × {target} — Cross-Distribution"

        clean = data_work[[feature, group_col]].dropna().copy()
        n_total = len(clean)

        value_counts = clean[feature].value_counts()
        if len(value_counts) > top_n:
            top_cats = set(value_counts.index[:top_n])
            clean[feature] = (
                clean[feature]
                .astype(object)
                .where(
                    clean[feature].isin(top_cats),
                    other=f"Other ({len(value_counts) - top_n})",
                )
            )

        contingency = pd.crosstab(clean[feature], clean[group_col])

        rows = []
        for cat in contingency.index:
            row: dict = {"Category": cat}
            total_count = int(contingency.loc[cat].sum())
            row["Count"] = total_count
            row["Count %"] = total_count / n_total * 100
            for cls in classes:
                cls_count = (
                    int(contingency.at[cat, cls]) if cls in contingency.columns else 0
                )
                row[f"{cls} Count"] = cls_count
                row[f"{cls} Row %"] = (
                    cls_count / total_count * 100 if total_count > 0 else 0.0
                )
            rows.append(row)

        df = (
            pd.DataFrame(rows)
            .sort_values("Count", ascending=False)
            .reset_index(drop=True)
        )

        total_row: dict = {"Category": "Total", "Count": n_total, "Count %": 100.0}
        for cls in classes:
            cls_total = int(contingency[cls].sum()) if cls in contingency.columns else 0
            total_row[f"{cls} Count"] = cls_total
            total_row[f"{cls} Row %"] = cls_total / n_total * 100
        df = pd.concat([df, pd.DataFrame([total_row])], ignore_index=True)

        format_dict: dict[str, str] = {"Count": "{:,.0f}", "Count %": "{:.2f}%"}
        for cls in classes:
            format_dict[f"{cls} Count"] = "{:,.0f}"
            format_dict[f"{cls} Row %"] = "{:.2f}%"

        def _highlight_total(row: pd.Series) -> list[str]:
            """Highlights the Total summary row."""
            if str(row["Category"]) != "Total":
                return [""] * len(row)
            return [
                "background-color: #d4e6f1; font-weight: bold; border-top: 2px solid #1e3a5f"
            ] * len(row)

        row_pct_cols = [f"{cls} Row %" for cls in classes]

        table_styles = [
            {
                "selector": "caption",
                "props": [
                    ("font-family", "Times New Roman"),
                    ("font-size", "1.7em"),
                    ("font-weight", "bold"),
                    ("color", "#1e3a5f"),
                    ("padding", "14px 0 10px 4px"),
                    ("text-align", "left"),
                    ("letter-spacing", "0.04em"),
                ],
            },
            {
                "selector": "thead th",
                "props": [
                    ("background-color", "#1e3a5f"),
                    ("color", "white"),
                    ("font-family", "Times New Roman"),
                    ("font-size", "1.3em"),
                    ("font-weight", "bold"),
                    ("padding", "11px 16px"),
                    ("border", "none"),
                    ("text-align", "center"),
                    ("letter-spacing", "0.04em"),
                    ("white-space", "nowrap"),
                ],
            },
            {
                "selector": "tbody tr:nth-child(even)",
                "props": [("background-color", "#eef3fb")],
            },
            {
                "selector": "tbody tr:nth-child(odd)",
                "props": [("background-color", "#ffffff")],
            },
            {
                "selector": "tbody tr:hover",
                "props": [("background-color", "#d0e4f7")],
            },
            {
                "selector": "",
                "props": [
                    ("border-collapse", "collapse"),
                    ("box-shadow", "0 4px 14px rgba(0,0,0,0.10)"),
                    ("width", "100%"),
                ],
            },
        ]

        styled = (
            df.style.set_caption(caption)
            .format(format_dict)
            .background_gradient(
                subset=row_pct_cols, cmap="Blues", axis=0, vmin=0, vmax=100
            )
            .apply(_highlight_total, axis=1)
            .set_table_styles(table_styles)
            .set_properties(
                **{
                    "font-family": "Times New Roman",
                    "font-size": "1.4em",
                    "padding": "9px 16px",
                    "border": "none",
                    "border-bottom": "1px solid #dde6f0",
                    "text-align": "center",
                    "color": "#2c3e50",
                }
            )
            .set_properties(
                subset=["Category"], **{"text-align": "left", "font-weight": "bold"}
            )
        )
        return styled

    @staticmethod
    def categorical_target_association(
        data: pd.DataFrame,
        categorical_features: list[str],
        target: str,
        problem_type: str,
        n_quantiles: int = 4,
    ) -> pd.DataFrame:
        """Ranked association between categorical features and a target.

        For every categorical column computes:

        - **Cramér's V**: symmetric, normalised chi-squared effect size ranging 0–1.
          Thresholds: negligible < 0.05, weak < 0.15, moderate < 0.30, strong ≥ 0.30.
        - **Theil's U (uncertainty coefficient)**: asymmetric — ``U(target|feature)``
          is the fraction of the target's Shannon entropy explained by knowing the
          feature's value.  Directly interpretable as predictive power for the target.
        - **χ² statistic** and **p-value** from Pearson's chi-squared test of
          independence.

        For ``problem_type="regression"`` the continuous target is first binned into
        ``n_quantiles`` equal-frequency quantile bands via ``_bin_regression_target``;
        the bands are treated as the target classes for all metrics above.

        Results are sorted by Cramér's V descending so the most discriminative
        features appear first.  Both Cramér's V and Theil's U are colour-coded
        (grey → amber → orange → green) and the p-value uses a significance gradient.

        Args:
            data (pd.DataFrame): The input DataFrame.
            categorical_features (list[str]): List of categorical feature names to analyze.
            target (str): Target column name.
            problem_type (str): Either ``"classification"`` or ``"regression"``.
            n_quantiles (int, optional): Number of quantile bands used when
                ``problem_type="regression"``. Defaults to 4.

        Returns:
            pd.DataFrame: A styled DataFrame sorted by Cramér's V descending.

        Raises:
            ValueError: If ``problem_type`` is invalid or ``target`` is absent.
        """
        if problem_type not in {"classification", "regression"}:
            raise ValueError(
                "problem_type must be either 'classification' or 'regression'."
            )
        if target not in data.columns:
            raise ValueError(f"Target column '{target}' not found in DataFrame.")

        if problem_type == "regression":
            data_work, _ = DataAnalyzer._bin_regression_target(
                data, target, n_quantiles
            )
            group_col = "__target_bin__"
            caption = (
                f"🔗 Categorical Features — Association with {target} (Quantile Bins)"
            )
        else:
            data_work = data
            group_col = target
            caption = f"🔗 Categorical Features — Association with {target}"

        rows = []
        for col in categorical_features:
            clean = data_work[[col, group_col]].dropna()
            if len(clean) == 0 or clean[col].nunique() < 2:
                continue
            v, chi2, p_val = DataAnalyzer._compute_cramers_v_chi2(
                clean[col], clean[group_col]
            )
            u = DataAnalyzer._compute_theil_u(clean[col], clean[group_col])
            rows.append(
                {
                    "Feature": col,
                    "N Unique": int(clean[col].nunique()),
                    "Cramér's V": v,
                    "Theil's U": u,
                    "χ²": chi2,
                    "p-value": p_val,
                }
            )

        df = (
            pd.DataFrame(rows)
            .sort_values("Cramér's V", ascending=False)
            .reset_index(drop=True)
        )

        table_styles = [
            {
                "selector": "caption",
                "props": [
                    ("font-family", "Times New Roman"),
                    ("font-size", "1.7em"),
                    ("font-weight", "bold"),
                    ("color", "#1e3a5f"),
                    ("padding", "14px 0 10px 4px"),
                    ("text-align", "left"),
                    ("letter-spacing", "0.04em"),
                ],
            },
            {
                "selector": "thead th",
                "props": [
                    ("background-color", "#1e3a5f"),
                    ("color", "white"),
                    ("font-family", "Times New Roman"),
                    ("font-size", "1.3em"),
                    ("font-weight", "bold"),
                    ("padding", "11px 16px"),
                    ("border", "none"),
                    ("text-align", "center"),
                    ("letter-spacing", "0.04em"),
                    ("white-space", "nowrap"),
                ],
            },
            {
                "selector": "tbody tr:nth-child(even)",
                "props": [("background-color", "#eef3fb")],
            },
            {
                "selector": "tbody tr:nth-child(odd)",
                "props": [("background-color", "#ffffff")],
            },
            {
                "selector": "tbody tr:hover",
                "props": [("background-color", "#d0e4f7")],
            },
            {
                "selector": "",
                "props": [
                    ("border-collapse", "collapse"),
                    ("box-shadow", "0 4px 14px rgba(0,0,0,0.10)"),
                    ("width", "100%"),
                ],
            },
        ]

        styled = (
            df.style.set_caption(caption)
            .format(
                {
                    "N Unique": "{:,.0f}",
                    "Cramér's V": "{:.5f}",
                    "Theil's U": "{:.5f}",
                    "χ²": "{:,.3f}",
                    "p-value": "{:.3e}",
                }
            )
            .map(
                DataAnalyzer._association_cell_color, subset=["Cramér's V", "Theil's U"]
            )
            .map(DataAnalyzer._pvalue_cell_color, subset=["p-value"])
            .set_table_styles(table_styles)
            .set_properties(
                **{
                    "font-family": "Times New Roman",
                    "font-size": "1.4em",
                    "padding": "9px 16px",
                    "border": "none",
                    "border-bottom": "1px solid #dde6f0",
                    "text-align": "center",
                    "color": "#2c3e50",
                }
            )
            .set_properties(
                subset=["Feature"], **{"text-align": "left", "font-weight": "bold"}
            )
        )
        return styled

    @staticmethod
    def _drift_score_cell_color(val: float) -> str:
        """Maps a drift score to a colour (green = stable, red = significant drift)."""
        if val < 0.05:
            r, g, b = 56, 142, 60
        elif val < 0.10:
            r, g, b = 255, 183, 77
        elif val < 0.20:
            r, g, b = 230, 81, 0
        else:
            r, g, b = 198, 40, 40
        return f"background-color: rgb({r},{g},{b}); color: white; font-weight: bold; text-align: center"

    @staticmethod
    def _drift_pvalue_cell_color(val: float) -> str:
        """Maps a drift-context p-value to colour: low p = drift detected = red."""
        if val < 0.001:
            r, g, b = 198, 40, 40
        elif val < 0.01:
            r, g, b = 230, 81, 0
        elif val < 0.05:
            r, g, b = 255, 183, 77
        elif val < 0.10:
            r, g, b = 104, 159, 56
        else:
            r, g, b = 56, 142, 60
        return f"background-color: rgb({r},{g},{b}); color: white; font-weight: bold; text-align: center"

    @staticmethod
    def _drift_flag_cell_color(flag: str) -> str:
        """Maps a drift flag text label to a background colour."""
        palette = {
            "STABLE": (56, 142, 60),
            "SLIGHT DRIFT": (255, 183, 77),
            "MODERATE DRIFT": (230, 81, 0),
            "SIGNIFICANT DRIFT": (198, 40, 40),
        }
        r, g, b = palette.get(flag, (189, 189, 189))
        return f"background-color: rgb({r},{g},{b}); color: white; font-weight: bold; text-align: center"

    @staticmethod
    def _compute_numerical_psi(
        train: np.ndarray, test: np.ndarray, n_bins: int = 10
    ) -> float:
        """PSI for a numerical feature using equal-frequency bins derived from train."""
        breakpoints = np.unique(np.percentile(train, np.linspace(0, 100, n_bins + 1)))
        if len(breakpoints) < 3:
            return np.nan
        bins = np.concatenate([[-np.inf], breakpoints[1:-1], [np.inf]])
        train_pcts = np.maximum(
            np.histogram(train, bins=bins)[0].astype(float) / len(train), 1e-6
        )
        test_pcts = np.maximum(
            np.histogram(test, bins=bins)[0].astype(float) / len(test), 1e-6
        )
        return float(np.sum((test_pcts - train_pcts) * np.log(test_pcts / train_pcts)))

    @staticmethod
    def _compute_categorical_psi(train: pd.Series, test: pd.Series) -> float:
        """PSI for a categorical feature, including new / missing category penalties."""
        all_cats = set(train.dropna().unique()) | set(test.dropna().unique())
        n_tr = max(len(train.dropna()), 1)
        n_te = max(len(test.dropna()), 1)
        psi = 0.0
        for cat in all_cats:
            p = max(float((train == cat).sum()) / n_tr, 1e-6)
            q = max(float((test == cat).sum()) / n_te, 1e-6)
            psi += (q - p) * np.log(q / p)
        return float(psi)

    @staticmethod
    def _compute_js_distance(p: np.ndarray, q: np.ndarray) -> float:
        """Jensen-Shannon distance (square root of JS divergence) for two distributions."""
        p = np.maximum(p / p.sum(), 1e-300)
        q = np.maximum(q / q.sum(), 1e-300)
        m = 0.5 * (p + q)
        kl_pm = float(np.sum(p * np.log(p / m)))
        kl_qm = float(np.sum(q * np.log(q / m)))
        return float(np.sqrt(max(0.5 * (kl_pm + kl_qm), 0.0)))

    @staticmethod
    def numerical_drift_analysis(
        train_data: pd.DataFrame,
        test_data: pd.DataFrame,
        numerical_features: list[str] | None = None,
        n_bins: int = 10,
    ) -> pd.DataFrame:
        """Distribution drift analysis for numerical features between train and test sets.

        For each numerical feature computes a battery of complementary drift metrics:

        - **KS Stat**: Kolmogorov-Smirnov two-sample statistic — maximum absolute
          difference between empirical CDFs, range [0, 1].
        - **KS p-value**: p-value of the KS test (low = distributions significantly
          differ; coloured red to signal detected drift).
        - **Wasserstein**: Earth Mover's Distance in the original feature units —
          the minimum cost of transforming one distribution into the other.  Sensitive
          to the magnitude of the shift.
        - **SMD**: Standardized Mean Difference ``(μ_test − μ_train) / σ_pooled`` —
          signed, scale-free measure of location shift.
        - **PSI**: Population Stability Index using ``n_bins`` equal-frequency train
          bins.  Thresholds: < 0.10 stable, 0.10–0.20 moderate, > 0.20 significant.
        - **Mean Δ%**: relative shift of the test mean compared to the train mean.
        - **Std Ratio**: ``σ_test / σ_train`` — detects dispersion changes.

        The **Flag** column combines KS stat and PSI into a four-level label:
        STABLE / SLIGHT DRIFT / MODERATE DRIFT / SIGNIFICANT DRIFT.

        Results are sorted by KS Stat descending so the most drifted features
        appear first.

        Args:
            train_data (pd.DataFrame): Reference (training) dataset.
            test_data (pd.DataFrame): Evaluation (test / production) dataset.
            numerical_features (list[str] | None, optional): Subset of numerical
                columns to analyse.  Auto-detects all shared numeric columns when
                ``None``. Defaults to None.
            n_bins (int, optional): Number of equal-frequency bins used for PSI.
                Defaults to 10.

        Returns:
            pd.DataFrame: A styled DataFrame sorted by KS Stat descending.
        """
        if numerical_features is None:
            numerical_features = [
                c
                for c in train_data.select_dtypes(include=[np.number]).columns
                if c in test_data.columns
            ]

        def _flag(ks_stat: float, psi: float) -> str:
            """Derives drift flag from KS statistic and PSI."""
            eff_psi = psi if not np.isnan(psi) else 0.0
            if ks_stat >= 0.20 or eff_psi >= 0.25:
                return "SIGNIFICANT DRIFT"
            if ks_stat >= 0.10 or eff_psi >= 0.20:
                return "MODERATE DRIFT"
            if ks_stat >= 0.05 or eff_psi >= 0.10:
                return "SLIGHT DRIFT"
            return "STABLE"

        rows = []
        for col in numerical_features:
            tr = train_data[col].dropna().values
            te = test_data[col].dropna().values
            if len(tr) < 2 or len(te) < 2:
                continue

            ks_stat, ks_p = ks_2samp(tr, te)
            w_dist = wasserstein_distance(tr, te)

            tr_mean, te_mean = float(tr.mean()), float(te.mean())
            tr_std, te_std = float(tr.std()), float(te.std())
            pooled_std = float(np.sqrt((tr_std**2 + te_std**2) / 2))
            smd = (te_mean - tr_mean) / pooled_std if pooled_std > 0 else 0.0
            mean_delta_pct = (
                (te_mean - tr_mean) / abs(tr_mean) * 100 if tr_mean != 0 else np.nan
            )
            std_ratio = te_std / tr_std if tr_std > 0 else np.nan
            psi = DataAnalyzer._compute_numerical_psi(tr, te, n_bins)

            rows.append(
                {
                    "Feature": col,
                    "KS Stat": ks_stat,
                    "KS p-value": ks_p,
                    "Wasserstein": w_dist,
                    "SMD": smd,
                    "PSI": psi,
                    "Mean Δ%": mean_delta_pct,
                    "Std Ratio": std_ratio,
                    "Flag": _flag(ks_stat, psi),
                }
            )

        df = (
            pd.DataFrame(rows)
            .sort_values("KS Stat", ascending=False)
            .reset_index(drop=True)
        )

        table_styles = [
            {
                "selector": "caption",
                "props": [
                    ("font-family", "Times New Roman"),
                    ("font-size", "1.7em"),
                    ("font-weight", "bold"),
                    ("color", "#1e3a5f"),
                    ("padding", "14px 0 10px 4px"),
                    ("text-align", "left"),
                    ("letter-spacing", "0.04em"),
                ],
            },
            {
                "selector": "thead th",
                "props": [
                    ("background-color", "#1e3a5f"),
                    ("color", "white"),
                    ("font-family", "Times New Roman"),
                    ("font-size", "1.3em"),
                    ("font-weight", "bold"),
                    ("padding", "11px 16px"),
                    ("border", "none"),
                    ("text-align", "center"),
                    ("letter-spacing", "0.04em"),
                    ("white-space", "nowrap"),
                ],
            },
            {
                "selector": "tbody tr:nth-child(even)",
                "props": [("background-color", "#eef3fb")],
            },
            {
                "selector": "tbody tr:nth-child(odd)",
                "props": [("background-color", "#ffffff")],
            },
            {
                "selector": "tbody tr:hover",
                "props": [("background-color", "#d0e4f7")],
            },
            {
                "selector": "",
                "props": [
                    ("border-collapse", "collapse"),
                    ("box-shadow", "0 4px 14px rgba(0,0,0,0.10)"),
                    ("width", "100%"),
                ],
            },
        ]

        styled = (
            df.style.set_caption("📉 Numerical Feature Drift Analysis")
            .format(
                {
                    "KS Stat": "{:.5f}",
                    "KS p-value": "{:.3e}",
                    "Wasserstein": "{:.5f}",
                    "SMD": "{:.5f}",
                    "PSI": "{:.5f}",
                    "Mean Δ%": "{:.2f}%",
                    "Std Ratio": "{:.3f}",
                },
                na_rep="—",
            )
            .map(DataAnalyzer._drift_score_cell_color, subset=["KS Stat", "PSI"])
            .map(DataAnalyzer._drift_pvalue_cell_color, subset=["KS p-value"])
            .map(DataAnalyzer._drift_flag_cell_color, subset=["Flag"])
            .set_table_styles(table_styles)
            .set_properties(
                **{
                    "font-family": "Times New Roman",
                    "font-size": "1.4em",
                    "padding": "9px 16px",
                    "border": "none",
                    "border-bottom": "1px solid #dde6f0",
                    "text-align": "center",
                    "color": "#2c3e50",
                }
            )
            .set_properties(
                subset=["Feature"], **{"text-align": "left", "font-weight": "bold"}
            )
            .set_properties(
                subset=["Flag"],
                **{"text-align": "left", "white-space": "nowrap"},
            )
        )
        return styled

    @staticmethod
    def categorical_drift_analysis(
        train_data: pd.DataFrame,
        test_data: pd.DataFrame,
        categorical_features: list[str] | None = None,
    ) -> pd.DataFrame:
        """Distribution drift analysis for categorical features between train and test sets.

        For each categorical feature computes:

        - **JS Distance**: Jensen-Shannon distance — symmetric, bounded [0, 1], measures
          the overall shape difference between train and test category distributions.
          More robust than chi-squared for sparse categories.
        - **PSI**: Population Stability Index summed across all categories (including
          new / missing ones, which incur a heavy penalty).
        - **χ² p-value**: p-value of a chi-squared test of independence on the 2 × K
          contingency table (train vs test rows, K = shared categories).  Low = drift
          detected.
        - **New Categories**: count of categories present in test but absent in train
          (unseen by the model at training time — a hard input-drift signal).
        - **Missing Categories**: count of categories present in train but absent in
          test (may indicate undersampling or segment disappearance).

        The **Flag** column combines JS Distance and PSI into a four-level label:
        STABLE / SLIGHT DRIFT / MODERATE DRIFT / SIGNIFICANT DRIFT.

        Results are sorted by JS Distance descending.

        Args:
            train_data (pd.DataFrame): Reference (training) dataset.
            test_data (pd.DataFrame): Evaluation (test / production) dataset.
            categorical_features (list[str] | None, optional): Subset of categorical
                columns to analyse.  Auto-detects all shared object/category/bool
                columns when ``None``. Defaults to None.

        Returns:
            pd.DataFrame: A styled DataFrame sorted by JS Distance descending.
        """
        if categorical_features is None:
            shared = set(train_data.columns) & set(test_data.columns)
            categorical_features = [
                c
                for c in train_data.select_dtypes(
                    include=["object", "string", "category", "bool"]
                ).columns
                if c in shared
            ]

        def _flag(js_dist: float, psi: float) -> str:
            """Derives drift flag from JS distance and PSI."""
            eff_psi = psi if not np.isnan(psi) else 0.0
            if js_dist >= 0.20 or eff_psi >= 0.25:
                return "SIGNIFICANT DRIFT"
            if js_dist >= 0.10 or eff_psi >= 0.20:
                return "MODERATE DRIFT"
            if js_dist >= 0.05 or eff_psi >= 0.10:
                return "SLIGHT DRIFT"
            return "STABLE"

        rows = []
        for col in categorical_features:
            tr = train_data[col].dropna()
            te = test_data[col].dropna()
            if len(tr) == 0 or len(te) == 0:
                continue

            train_cats = set(tr.unique())
            test_cats = set(te.unique())
            all_cats = sorted(train_cats | test_cats)
            n_tr, n_te = len(tr), len(te)

            p_train = np.array([(tr == c).sum() / n_tr for c in all_cats], dtype=float)
            p_test = np.array([(te == c).sum() / n_te for c in all_cats], dtype=float)
            js_dist = DataAnalyzer._compute_js_distance(p_train, p_test)
            psi = DataAnalyzer._compute_categorical_psi(tr, te)

            common_cats = sorted(train_cats & test_cats)
            if len(common_cats) >= 2:
                counts_tr = np.array([(tr == c).sum() for c in common_cats])
                counts_te = np.array([(te == c).sum() for c in common_cats])
                _, chi2_p, _, _ = chi2_contingency(np.array([counts_tr, counts_te]))
            else:
                chi2_p = np.nan

            rows.append(
                {
                    "Feature": col,
                    "JS Distance": js_dist,
                    "PSI": psi,
                    "χ² p-value": chi2_p,
                    "New Categories": len(test_cats - train_cats),
                    "Missing Categories": len(train_cats - test_cats),
                    "Flag": _flag(js_dist, psi),
                }
            )

        df = (
            pd.DataFrame(rows)
            .sort_values("JS Distance", ascending=False)
            .reset_index(drop=True)
        )

        table_styles = [
            {
                "selector": "caption",
                "props": [
                    ("font-family", "Times New Roman"),
                    ("font-size", "1.7em"),
                    ("font-weight", "bold"),
                    ("color", "#1e3a5f"),
                    ("padding", "14px 0 10px 4px"),
                    ("text-align", "left"),
                    ("letter-spacing", "0.04em"),
                ],
            },
            {
                "selector": "thead th",
                "props": [
                    ("background-color", "#1e3a5f"),
                    ("color", "white"),
                    ("font-family", "Times New Roman"),
                    ("font-size", "1.3em"),
                    ("font-weight", "bold"),
                    ("padding", "11px 16px"),
                    ("border", "none"),
                    ("text-align", "center"),
                    ("letter-spacing", "0.04em"),
                    ("white-space", "nowrap"),
                ],
            },
            {
                "selector": "tbody tr:nth-child(even)",
                "props": [("background-color", "#eef3fb")],
            },
            {
                "selector": "tbody tr:nth-child(odd)",
                "props": [("background-color", "#ffffff")],
            },
            {
                "selector": "tbody tr:hover",
                "props": [("background-color", "#d0e4f7")],
            },
            {
                "selector": "",
                "props": [
                    ("border-collapse", "collapse"),
                    ("box-shadow", "0 4px 14px rgba(0,0,0,0.10)"),
                    ("width", "100%"),
                ],
            },
        ]

        styled = (
            df.style.set_caption("📉 Categorical Feature Drift Analysis")
            .format(
                {
                    "JS Distance": "{:.5f}",
                    "PSI": "{:.5f}",
                    "χ² p-value": "{:.3e}",
                    "New Categories": "{:,.0f}",
                    "Missing Categories": "{:,.0f}",
                },
                na_rep="—",
            )
            .map(DataAnalyzer._drift_score_cell_color, subset=["JS Distance", "PSI"])
            .map(DataAnalyzer._drift_pvalue_cell_color, subset=["χ² p-value"])
            .map(DataAnalyzer._drift_flag_cell_color, subset=["Flag"])
            .set_table_styles(table_styles)
            .set_properties(
                **{
                    "font-family": "Times New Roman",
                    "font-size": "1.4em",
                    "padding": "9px 16px",
                    "border": "none",
                    "border-bottom": "1px solid #dde6f0",
                    "text-align": "center",
                    "color": "#2c3e50",
                }
            )
            .set_properties(
                subset=["Feature"], **{"text-align": "left", "font-weight": "bold"}
            )
            .set_properties(
                subset=["Flag"],
                **{"text-align": "left", "white-space": "nowrap"},
            )
        )
        return styled

    @staticmethod
    def _prepare_features_for_mutual_information(
        X: pd.DataFrame,
    ) -> tuple[pd.DataFrame, np.ndarray | None]:
        """Prepare feature matrix for mutual information computation."""
        if not isinstance(X, pd.DataFrame):
            raise ValueError("X must be a pandas DataFrame.")

        discrete_columns = X.select_dtypes(
            include=["category", "object", "bool"]
        ).columns
        if len(discrete_columns) == 0:
            return X, None

        X_prepared = X.copy()
        for column in discrete_columns:
            X_prepared[column] = pd.Categorical(X_prepared[column]).codes

        discrete_features_mask = X_prepared.columns.isin(discrete_columns).astype(bool)
        return X_prepared, discrete_features_mask

    @staticmethod
    def mutual_information_scores(
        X: pd.DataFrame,
        y: pd.Series | np.ndarray,
        problem_type: str,
        n_neighbors: int = 3,
        random_state: int | None = 17,
    ) -> dict[str, float]:
        """Compute descending mutual information scores for all features.

        Args:
            X (pd.DataFrame): Feature matrix.
            y (pd.Series | np.ndarray): Target variable aligned with X.
            problem_type (str): Problem type, either "classification" or "regression".
            n_neighbors (int, optional): Number of neighbors used by MI estimators.
                Defaults to 3.
            random_state (int | None, optional): Random state for reproducibility.
                Defaults to 17.

        Returns:
            dict[str, float]: Ordered mapping from feature name to MI score sorted
                from highest to lowest.

        Raises:
            ValueError: If input types are invalid or problem_type is unsupported.

        Note:
            Time complexity is dominated by sklearn MI estimation and is approximately
            O(n_samples * n_features) for fixed n_neighbors; additional sorting is
            O(n_features * log(n_features)).
        """
        if not isinstance(y, (pd.Series, np.ndarray)):
            raise ValueError("y must be a pandas Series or numpy array.")
        if problem_type not in {"classification", "regression"}:
            raise ValueError(
                "problem_type must be either 'classification' or 'regression'."
            )

        X_prepared, discrete_features_mask = (
            DataAnalyzer._prepare_features_for_mutual_information(X)
        )

        mi_kwargs: dict[str, int | np.ndarray] = {"n_neighbors": n_neighbors}
        if random_state is not None:
            mi_kwargs["random_state"] = random_state
        if discrete_features_mask is not None:
            mi_kwargs["discrete_features"] = discrete_features_mask

        if problem_type == "classification":
            mi_scores = mutual_info_classif(X_prepared, y, **mi_kwargs)
        else:
            mi_scores = mutual_info_regression(X_prepared, y, **mi_kwargs)

        mi_scores = np.nan_to_num(mi_scores, nan=0.0, posinf=0.0, neginf=0.0)
        sorted_indices = np.argsort(mi_scores)[::-1]

        return {
            str(X_prepared.columns[idx]): float(mi_scores[idx])
            for idx in sorted_indices
        }


if __name__ == "__main__":
    # Create sample data for demonstration
    sample_data = pd.DataFrame(
        {
            "numerical_feature": np.random.normal(100, 15, 1000),
            "categorical_feature": np.random.choice(["A", "B", "C", "D"], 1000),
            "date_feature": pd.date_range("2023-01-01", periods=1000, freq="D"),
        }
    )

    # Add some missing values
    sample_data.loc[50:60, "numerical_feature"] = np.nan
    sample_data.loc[100:110, "categorical_feature"] = np.nan

    # Demonstrate DataAnalyzer usage
    data_analyzer = DataAnalyzer()

    # Get base information
    base_info = data_analyzer.base_information(sample_data)
    print("Base Information:")
    print(base_info.data)

    # Get numerical feature summary
    categorical_features = data_analyzer.get_categorical_features(sample_data)
    print("\nCategorical Features:")
    print(categorical_features)

    numerical_feature_summary = data_analyzer.describe_numerical_feature(
        sample_data, "numerical_feature"
    )
    print("\nNumerical Feature Summary:")
    print(numerical_feature_summary.data)
