import pandas as pd
import pytest

from kvbiii_ml.data_processing.feature_engineering.case_normalizer import CaseNormalizer


def test_case_normalizer_lower_object_and_string() -> None:
    """Lower normalization should handle object and string dtypes."""
    df = pd.DataFrame(
        {
            "obj": ["A", "b", None],
            "str": pd.Series(["X", "y", pd.NA], dtype="string"),
            "num": [1, 2, 3],
        }
    )

    normalizer = CaseNormalizer(features_names=["obj", "str"], normalization="lower")
    normalizer.fit(df)
    out = normalizer.transform(df)

    if not (out.loc[0, "obj"] == "a"):
        raise AssertionError()
    if not (out.loc[1, "obj"] == "b"):
        raise AssertionError()
    if not (pd.isna(out.loc[2, "obj"])):
        raise AssertionError()
    if not (out.loc[0, "str"] == "x"):
        raise AssertionError()
    if not (out.loc[1, "str"] == "y"):
        raise AssertionError()
    if not (pd.isna(out.loc[2, "str"])):
        raise AssertionError()
    if not (out["num"].tolist() == [1, 2, 3]):
        raise AssertionError()


def test_case_normalizer_preserves_non_string_values() -> None:
    """Non-string values should remain unchanged during normalization."""
    df = pd.DataFrame({"mixed": ["A", 1, 2.5, None]})

    normalizer = CaseNormalizer(features_names=["mixed"], normalization="upper")
    normalizer.fit(df)
    out = normalizer.transform(df)

    if not (out.loc[0, "mixed"] == "A"):
        raise AssertionError()
    if not (out.loc[1, "mixed"] == 1):
        raise AssertionError()
    if not (out.loc[2, "mixed"] == 2.5):
        raise AssertionError()
    if not (pd.isna(out.loc[3, "mixed"])):
        raise AssertionError()


def test_case_normalizer_auto_detects_string_like_columns() -> None:
    """Auto-detection should select object, string, and category columns."""
    df = pd.DataFrame(
        {
            "cat": pd.Series(["UP", "Down"], dtype="category"),
            "obj": ["Left", "Right"],
            "num": [10, 20],
        }
    )

    normalizer = CaseNormalizer(normalization="lower")
    normalizer.fit(df)
    out = normalizer.transform(df)

    if not (normalizer.features_names_ == ["cat", "obj"]):
        raise AssertionError()
    if not (out["cat"].dtype.name == "category"):
        raise AssertionError()
    if not (list(out["cat"].cat.categories) == ["up", "down"]):
        raise AssertionError()
    if not (out["obj"].tolist() == ["left", "right"]):
        raise AssertionError()


def test_case_normalizer_invalid_normalization_raises_error() -> None:
    """Invalid normalization values should raise a ValueError."""
    with pytest.raises(ValueError, match="normalization must be one of"):
        CaseNormalizer(normalization="invalid")


def test_case_normalizer_get_feature_names_out_returns_inputs() -> None:
    """Feature names output should match original input columns."""
    df = pd.DataFrame({"A": ["x"], "B": [1]})
    normalizer = CaseNormalizer(features_names=["A"])

    normalizer.fit(df)

    if not (list(normalizer.get_feature_names_out()) == ["A", "B"]):
        raise AssertionError()


if __name__ == "__main__":
    print("Run this file with pytest to execute tests.")
