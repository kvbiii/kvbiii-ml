from sklearn.exceptions import NotFittedError


def check_fitted(obj: object, *attributes: str, method: str = "fit") -> None:
    """
    Raise NotFittedError unless ``method`` has set every attribute on ``obj``.

    sklearn's ``check_is_fitted`` needs a ``BaseEstimator``; this works on any class.

    Args:
        obj (object): Instance to check.
        *attributes (str): Attribute names that only exist after ``method`` ran.
        method (str): Name of the method that sets them, used in the message. Defaults to "fit".

    Raises:
        NotFittedError: If any attribute is missing.
    """
    if not all(hasattr(obj, name) for name in attributes):
        msg = f"This {type(obj).__name__} instance is not fitted yet. Call {method}() first."
        raise NotFittedError(msg)
