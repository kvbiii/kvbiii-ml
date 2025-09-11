import itertools
import numpy as np
import pandas as pd
try:  # optional dependency
    from kvbiii_plots.eda.multivariate_plots import MultivariatePlots  # type: ignore
except Exception:  # pragma: no cover
    class MultivariatePlots:  # type: ignore
        def __init__(self, *_, **__):
            pass
        def heatmap(self, *_, **__):
            return None

multivariate_plots = MultivariatePlots()


def impute_missing_values(
    df: pd.DataFrame,
    categorical_to_impute: list[str],
    non_missing_categorical: list[str],
    threshold_num_observation: int = 10,
) -> pd.DataFrame:
    """Impute missing values using joint distributions of categorical pairs.

    For each pair (to_impute, by_column), if a category in by_column maps to a
    single non-zero category in to_impute and has at least the given threshold
    of observations, the missing values in to_impute are filled with that unique
    category.

    Args:
        df (pd.DataFrame): Input DataFrame.
        categorical_to_impute (list[str]): Categorical columns with missing values.
        non_missing_categorical (list[str]): Categorical columns with no missing values.
        threshold_num_observation (int, optional): Minimum observations required per by-category. Defaults to 10.

    Returns:
        pd.DataFrame: A copy of the DataFrame with imputed values.
    """
    df = df.copy()
    for col_impute, col_by in itertools.product(
        categorical_to_impute, non_missing_categorical
    ):
        # Cross-tab counts excluding NaNs in the column being imputed
        joint_distribution = (
            df.groupby([col_impute, col_by], observed=False)[col_by]
            .size()
            .unstack()
            .fillna(0)
        )

        # Identify by-categories that map to exactly one non-zero category of col_impute
        nonzero_counts = (joint_distribution != 0.0).sum(axis=0)
        col_sums = joint_distribution.sum(axis=0)
        eligible_cols = nonzero_counts.eq(1) & col_sums.ge(threshold_num_observation)
        if not eligible_cols.any():
            continue

        # Build mapping: by_value -> impute_value
        mapping: dict = {}
        for by_val in joint_distribution.columns[eligible_cols]:
            a_vals = joint_distribution.index[joint_distribution[by_val] > 0]
            if len(a_vals) == 1:
                mapping[by_val] = a_vals[0]

        if not mapping:
            continue

        # Vectorized imputation for all eligible by-values
        mask = df[col_impute].isna() & df[col_by].isin(mapping)
        if mask.any():
            by_vals_to_fill = df.loc[mask, col_by].unique().tolist()
            highlights = [(mapping[bv], bv) for bv in by_vals_to_fill if bv in mapping]
            multivariate_plots.heatmap(
                data=joint_distribution,
                xaxis_title=col_impute,
                yaxis_title=col_by,
                plot_title=f"Imputation map: {col_impute} by {col_by}",
                highlights=highlights,
            )

            filled_values = df.loc[mask, col_by].map(mapping)
            df.loc[mask, col_impute] = filled_values

            # Per-category logging consistent with original intent
            for by_val, impute_val in mapping.items():
                submask = mask & (df[col_by] == by_val)
                count = int(submask.sum())
                if count > 0:
                    print(
                        f"Imputed {count} rows for column {col_impute} with value: {impute_val} "
                        f"based on {col_by} column with value: {by_val}."
                    )
    return df


impute_with_joint_distribution = impute_missing_values  # pragma: no cover alias


if __name__ == "__main__":
    # Minimal usage example ensuring deterministic imputation occurs
    rng = np.random.default_rng(17)

    n_u, n_v, n_w = 30, 25, 20
    df_u = pd.DataFrame(
        {
            "A": ["x"] * n_u,
            "B": ["u"] * n_u,
            "C": rng.choice(["m", "n"], size=n_u),
        }
    )
    # Introduce missing A for some rows with B = 'u'
    miss_idx_u = rng.choice(n_u, size=8, replace=False)
    df_u.loc[miss_idx_u, "A"] = None

    df_v = pd.DataFrame(
        {
            "A": ["y"] * n_v,
            "B": ["v"] * n_v,
            "C": rng.choice(["m", "n"], size=n_v),
        }
    )
    # Introduce missing A for some rows with B = 'v'
    miss_idx_v = rng.choice(n_v, size=5, replace=False)
    df_v.loc[miss_idx_v, "A"] = None

    df_w = pd.DataFrame(
        {
            "A": rng.choice(["x", "y", "z"], size=n_w),
            "B": ["w"] * n_w,
            "C": rng.choice(["m", "n"], size=n_w),
        }
    )

    df_demo = pd.concat([df_u, df_v, df_w], ignore_index=True)
    # Shuffle rows for realism
    df_demo = df_demo.sample(frac=1.0, random_state=17).reset_index(drop=True)
    # Cast to category dtype
    for col in ["A", "B", "C"]:
        df_demo[col] = df_demo[col].astype("category")

    print("Nulls in A before:", int(df_demo["A"].isna().sum()))
    df_imp = impute_missing_values(
        df_demo,
        categorical_to_impute=["A"],
        non_missing_categorical=["B"],
        threshold_num_observation=5,
    )
    print("Nulls in A after:", int(df_imp["A"].isna().sum()))
