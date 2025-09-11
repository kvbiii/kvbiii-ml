from __future__ import annotations

import numpy as np
import pandas as pd
try:  # optional interactive display
    from IPython.display import display  # type: ignore
except Exception:  # pragma: no cover
    def display(obj):  # type: ignore
        print(obj)
from scipy.stats import chi2_contingency, ks_2samp
from sklearn.feature_selection import mutual_info_classif


class StatisticalAssociationImputer:
    def __init__(
        self,
        metrics: list[str] = ["cramers_v", "mutual_info", "theils_u"],
        top_n: int = 2,
        verbose: bool = False,
    ) -> None:
        """
        Initialize the imputer.

        Args:
            metrics (list[str]): Association metrics to use for ranking. Defaults to ["cramers_v", "mutual_info", "theils_u"].
            top_n (int): Number of top-ranked columns to use for grouping. Defaults to 2.
            verbose (bool): Whether to print progress information. Defaults to False.
        """
        self.metrics = metrics
        self.top_n = top_n
        self.verbose = verbose
        self.rankings = None
        self.selected_cols = None
        self.bin_edges = {}
        self.categorical_columns = None
        self.impute_stats = None

    def fit(self, df: pd.DataFrame) -> StatisticalAssociationImputer:
        """
        Fit the imputer and prepare group statistics for all features.

        This learns bin edges for numeric columns, ranks features by association
        per target feature, selects grouping columns, and stores per-feature
        imputation statistics for later use in transform.

        Args:
            df (pd.DataFrame): Training data used to learn binning and imputation patterns.

        Returns:
            StatisticalAssociationImputer: Fitted instance for chaining.
        """
        self.bin_edges = {}
        self.categorical_columns = self._get_categorical_columns(df)

        all_features = [col for col in df.columns]
        if self.verbose:
            print(f"Preparing imputation statistics for all features: {all_features}")
        self.impute_stats = {}
        for feature in all_features:
            self.rank_metrics(df, feature)
            self.select_grouping_columns()
            self.impute_stats[feature] = {
                "stats": self.calculate_stats(df, feature),
                "selected_cols": self.selected_cols.copy(),
                "rankings": self.rankings.copy(),
            }

        return self

    def _get_categorical_columns(self, df: pd.DataFrame) -> list[str]:
        """
        Identify categorical-like columns with limited cardinality.

        Args:
            df (pd.DataFrame): Input dataframe.

        Returns:
            list[str]: Names of columns treated as categorical.
        """
        categorical_columns = df.select_dtypes(include=["object", "category"]).columns
        categorical_columns = [
            col for col in categorical_columns if df[col].nunique() < 100
        ]
        categorical_columns = list(
            set(categorical_columns) | set(df.columns[df.nunique() == 2])
        )
        return categorical_columns

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Impute missing values in new data using learned statistics.

        Args:
            df (pd.DataFrame): Data to impute.

        Returns:
            pd.DataFrame: Dataframe with missing values imputed.

        Raises:
            ValueError: If fit was not called before transform.
        """
        if self.impute_stats is None:
            raise ValueError("Must call fit() first to learn binning and grouping.")

        df = df.copy()
        columns_with_missing = df.columns[df.isnull().any()].tolist()

        if self.verbose and columns_with_missing:
            print(f"Found missing values in columns: {columns_with_missing}")
        elif self.verbose:
            print("No missing values found in any columns.")

        for feature in columns_with_missing:
            if feature not in self.impute_stats:
                if self.verbose:
                    print(
                        f"Warning: No imputation info for feature '{feature}', skipping."
                    )
                continue

            feature_info = self.impute_stats[feature]
            self.selected_cols = feature_info["selected_cols"]

            df = self.impute_with_stats(df, feature, feature_info["stats"])

            if self.verbose:
                self.ks_test(df, feature)
                # self.plot_distribution(df, feature)

            df[feature] = df["imputed"]
            df.drop(columns="imputed", inplace=True)

            if f"{feature}_imputed" in df.columns:
                df.drop(columns=f"{feature}_imputed", inplace=True)

        return df

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Fit the imputer and transform the data in one step.

        Args:
            df (pd.DataFrame): Data to fit on and then impute.

        Returns:
            pd.DataFrame: Imputed dataframe.
        """
        self.fit(df)
        return self.transform(df)

    def rank_metrics(self, df: pd.DataFrame, feature_name: str) -> pd.DataFrame:
        """
        Rank columns by association with the target feature.

        Args:
            df (pd.DataFrame): Input dataframe.
            feature_name (str): Target feature to evaluate associations against.

        Returns:
            pd.DataFrame: Rankings with metric scores and cumulative ranks.
        """
        from collections import defaultdict

        rankings = defaultdict(dict)
        df_bin = self.bin_dataframe(df, feature_name)

        if "cramers_v" in self.metrics:
            for col in df_bin.columns:
                if col != feature_name and df_bin[col].nunique() > 1:
                    rankings["cramers_v"][col] = self.cramers_v(
                        df_bin[col], df_bin[feature_name]
                    )

        if "mutual_info" in self.metrics:
            # Use the same row subset for all columns (as in original behavior), but
            # compute mutual information for all features in one batch for efficiency.
            temp = df_bin.dropna()
            if len(temp) > 0:
                temp_codes = temp.apply(lambda s: s.astype("category").cat.codes)
                y = temp_codes[feature_name].values
                X_codes = temp_codes.drop(columns=[feature_name])
                if X_codes.shape[1] > 0:
                    mi = mutual_info_classif(X_codes.values, y, discrete_features=True)
                    for i, col in enumerate(X_codes.columns):
                        rankings["mutual_info"][col] = float(mi[i])

        if "theils_u" in self.metrics:
            for col in df_bin.columns:
                if col != feature_name and df_bin[col].nunique() > 1:
                    rankings["theils_u"][col] = self.theils_u(
                        df_bin[col], df_bin[feature_name]
                    )

        df_ranks = pd.DataFrame(rankings)
        for metric in df_ranks.columns:
            df_ranks[f"{metric}_rank"] = df_ranks[metric].rank(ascending=False)
        df_ranks["total_rank"] = df_ranks[
            [c for c in df_ranks.columns if c.endswith("_rank")]
        ].sum(axis=1)
        df_ranks = df_ranks.sort_values("total_rank")
        if self.verbose:
            display(df_ranks)
        self.rankings = df_ranks
        return df_ranks

    def bin_dataframe(
        self, df: pd.DataFrame, feature_name: str, fit_bins: bool = True
    ) -> pd.DataFrame:
        """
        Bin numerical features to reduce cardinality for association metrics.

        Args:
            df (pd.DataFrame): Input dataframe.
            feature_name (str): Feature evaluated as the target (excluded from numeric binning set).
            fit_bins (bool): Whether to compute new bins. If False, reuse stored bins. Defaults to True.

        Returns:
            pd.DataFrame: Dataframe with binned numeric features.
        """
        df_bin = df.copy()
        numerical_columns = list(
            set(df.columns) - set(self.categorical_columns) - {feature_name}
        )
        for col in numerical_columns:
            try:
                non_na_data = df[col].dropna()
                if len(non_na_data) > 1:
                    # Compute bin edges once per numeric column; reuse thereafter to avoid recomputation.
                    if fit_bins and col not in self.bin_edges:
                        _, bin_edges = pd.qcut(
                            non_na_data, q=10, duplicates="drop", retbins=True
                        )
                        self.bin_edges[col] = bin_edges
                    if col in self.bin_edges:
                        df_bin[col] = pd.cut(
                            df[col],
                            bins=self.bin_edges[col],
                            include_lowest=True,
                            duplicates="drop",
                        )
            except (ValueError, TypeError):
                df_bin[col] = df[col]
        return df_bin

    @staticmethod
    def cramers_v(x: pd.Series, y: pd.Series) -> float:
        """
        Compute Cramer's V association between two categorical variables.

        Args:
            x (pd.Series): First categorical variable.
            y (pd.Series): Second categorical variable.

        Returns:
            float: Cramer's V in [0, 1]. Returns 0.0 if computation is not possible.
        """
        try:
            temp_df = pd.DataFrame({"x": x, "y": y}).dropna()
            if len(temp_df) < 2:
                return 0.0
            if temp_df["x"].nunique() <= 1 or temp_df["y"].nunique() <= 1:
                return 0.0
            confusion_matrix = pd.crosstab(temp_df["x"], temp_df["y"])
            chi2 = chi2_contingency(confusion_matrix)[0]
            n = confusion_matrix.sum().sum()
            if n == 0:
                return 0.0

            phi2 = chi2 / n
            r, k = confusion_matrix.shape
            phi2corr = max(0, phi2 - ((k - 1) * (r - 1)) / (n - 1))
            rcorr = r - ((r - 1) ** 2) / (n - 1)
            kcorr = k - ((k - 1) ** 2) / (n - 1)

            denominator = min((kcorr - 1), (rcorr - 1))
            if denominator <= 0:
                return 0.0

            return np.sqrt(phi2corr / denominator)
        except (ValueError, ZeroDivisionError, IndexError):
            return 0.0

    @staticmethod
    def theils_u(x: pd.Series, y: pd.Series) -> float:
        """
        Compute Theil's U uncertainty coefficient between two categorical variables.

        Args:
            x (pd.Series): First categorical variable.
            y (pd.Series): Second categorical variable.

        Returns:
            float: Theil's U in [0, 1]. Returns 0.0 if computation is not possible.
        """
        try:
            temp_df = pd.DataFrame({"x": x, "y": y}).dropna()
            if len(temp_df) < 2:
                return 0.0
            if temp_df["x"].nunique() <= 1 or temp_df["y"].nunique() <= 1:
                return 0.0
            s_xy = pd.crosstab(temp_df["x"], temp_df["y"])
            s_x = s_xy.sum(axis=1)
            s_y = s_xy.sum(axis=0)
            total = s_xy.sum().sum()
            if total == 0:
                return 0.0
            from math import log

            H_y = -sum(
                (val / total) * log(val / total + 1e-10) for val in s_y if val > 0
            )
            if H_y == 0:
                return 0.0
            H_y_given_x = 0
            for row in s_xy.itertuples(index=False):
                row_total = sum(row)
                if row_total == 0:
                    continue
                H_row = -sum(
                    (val / row_total) * log(val / row_total + 1e-10)
                    for val in row
                    if val > 0
                )
                H_y_given_x += row_total / total * H_row
            return (H_y - H_y_given_x) / H_y
        except (ValueError, ZeroDivisionError, IndexError):
            return 0.0

    def select_grouping_columns(self) -> list[str]:
        """
        Select top-N columns for grouping based on rankings.

        Returns:
            list[str]: Selected column names.

        Raises:
            ValueError: If rank_metrics was not called before selection.
        """
        if self.rankings is None:
            raise ValueError("Run rank_metrics first.")
        self.selected_cols = self.rankings.head(self.top_n).index.tolist()
        if self.verbose:
            print(f"Selected grouping columns: {self.selected_cols}")
        return self.selected_cols

    def calculate_stats(self, df: pd.DataFrame, feature_name: str) -> dict[str, object]:
        """
        Compute group-wise and overall statistics for a feature.

        Uses binned grouping columns for stability while computing stats on original values.

        Args:
            df (pd.DataFrame): Data containing the feature and grouping columns.
            feature_name (str): Feature to compute statistics for.

        Returns:
            dict[str, object]: Dictionary with keys "group" (mapping from grouped keys to stats)
            and "overall" (overall fallback statistic).
        """
        is_categorical = feature_name in self.categorical_columns
        df_binned = self.bin_dataframe(df, feature_name, fit_bins=False)

        df_for_stats = df_binned.copy()
        df_for_stats[feature_name] = df[feature_name]
        grouping_cols_binned = []
        for col in self.selected_cols:
            if col in df_binned.columns:
                grouping_cols_binned.append(col)

        grouped = df_for_stats.groupby(
            grouping_cols_binned, observed=True, dropna=False
        )[feature_name]

        if is_categorical:
            group_stats = grouped.agg(
                lambda x: x.mode().iloc[0] if not x.mode().empty else np.nan
            ).dropna()
            overall_stat = (
                df[feature_name].mode().iloc[0]
                if not df[feature_name].mode().empty
                else np.nan
            )
        else:
            group_stats = grouped.mean().dropna()
            overall_stat = df[feature_name].mean()

        if len(grouping_cols_binned) == 1:
            group_stats.index = [(val,) for val in group_stats.index]

        stat_dict = {"group": group_stats.to_dict(), "overall": overall_stat}
        if self.verbose:
            print("Stats by group (using binned grouping):")
            for group, stat in stat_dict["group"].items():
                group_str = ", ".join(
                    f"{col}: {val}" for col, val in zip(self.selected_cols, group)
                )
                print(f"{group_str} -> {stat}")
            print("Overall stat:", stat_dict["overall"])
        return stat_dict

    def impute_with_stats(
        self, df: pd.DataFrame, feature_name: str, impute_stats: dict[str, object]
    ) -> pd.DataFrame:
        """
        Impute missing values using precomputed group-wise statistics.

        Args:
            df (pd.DataFrame): Data containing the feature to impute.
            feature_name (str): Feature to impute.
            impute_stats (dict[str, object]): Mapping with "group" and "overall" statistics.

        Returns:
            pd.DataFrame: Dataframe with an additional "imputed" column before assignment to the feature.
        """
        missing_mask = df[feature_name].isna()
        df["imputed"] = df[feature_name]
        df[f"{feature_name}_imputed"] = False
        if not missing_mask.any():
            if self.verbose:
                print(f"No missing values to impute for '{feature_name}'.")
            return df

        # Bin once for consistency, then join group-wise stats in a vectorized way.
        df_binned = self.bin_dataframe(df, feature_name, fit_bins=False)

        # Build mapping DataFrame from group stats dict
        if len(self.selected_cols) == 1:
            # Keys are singletons like (val,), align with the single column
            mapping_df = pd.DataFrame(
                [(k[0], v) for k, v in impute_stats["group"].items()],
                columns=[self.selected_cols[0], "__group_stat__"],
            )
        else:
            mapping_index = pd.MultiIndex.from_tuples(
                list(impute_stats["group"].keys()), names=self.selected_cols
            )
            mapping_df = pd.Series(
                impute_stats["group"], index=mapping_index, name="__group_stat__"
            ).reset_index()

        to_impute = df_binned.loc[missing_mask, self.selected_cols].copy()
        joined = to_impute.merge(mapping_df, on=self.selected_cols, how="left")
        fill_values = joined["__group_stat__"].fillna(impute_stats["overall"]).values

        df.loc[missing_mask, "imputed"] = fill_values
        df.loc[missing_mask, f"{feature_name}_imputed"] = True
        if self.verbose:
            print(
                f"Imputed {missing_mask.sum()} missing values in '{feature_name}' using group-wise statistics."
            )
        return df

    def ks_test(self, df: pd.DataFrame, feature_name: str) -> tuple[float, float]:
        """
        Run a Kolmogorov–Smirnov test comparing original vs. imputed distributions.

        Args:
            df (pd.DataFrame): Data containing the feature and an "imputed" column.
            feature_name (str): Feature name under test.

        Returns:
            tuple[float, float]: KS statistic and p-value.
        """
        original = df[feature_name].dropna()
        imputed = df["imputed"]
        ks_stat, ks_p = ks_2samp(original, imputed)
        print(
            f"KS Statistic: {ks_stat:.4f}, p-value: {ks_p:.4f} - {'Distributions are significantly different (reject H0)' if ks_p < 0.05 else 'Distributions are not significantly different (fail to reject H0)'}"
        )
        return ks_stat, ks_p


if __name__ == "__main__":
    # Minimal runnable example
    import pandas as pd
    import numpy as np

    data = pd.DataFrame(
        {
            "A": ["x", "y", None, "x", "y", None, "x"],
            "B": [1.2, 2.3, 2.1, np.nan, 3.4, 1.0, 0.5],
            "C": ["u", "v", "u", "u", "v", "u", "v"],
        }
    )

    imputer = StatisticalAssociationImputer(verbose=True)
    imputer.fit(data)
    result = imputer.transform(data)
    print(result)
