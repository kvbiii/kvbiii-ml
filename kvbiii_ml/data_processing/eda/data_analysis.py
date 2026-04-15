from ast import literal_eval

import numpy as np
import pandas as pd
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
        categorical_features = df.select_dtypes(include=["object", "category"]).columns
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
    import numpy as np

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
    analyzer = DataAnalyzer()

    # Get base information
    base_info = analyzer.base_information(sample_data)
    print("Base Information:")
    print(base_info)

    # Analyze numerical feature
    numerical_stats = analyzer.describe_numerical_feature(
        sample_data, "numerical_feature"
    )
    print("\nNumerical Feature Analysis:")
    print(numerical_stats)

    # Analyze categorical feature
    categorical_stats = analyzer.describe_categorical_feature(
        sample_data, "categorical_feature"
    )
    print("\nCategorical Feature Analysis:")
    print(categorical_stats)

    # Get categorical features
    cat_cols = analyzer.get_categorical_features(sample_data)
    print(f"\nCategorical features: {cat_cols}")

    # Analyze time series feature
    time_stats = analyzer.describe_time_series_feature(
        sample_data, "date_feature", "ME"
    )
    print("\nTime Series Feature Analysis:")
    print(time_stats)
