from __future__ import annotations
import importlib
from importlib.metadata import PackageNotFoundError, version
from . import data_processing, evaluation, modeling

__version__ = "0+unknown"
try:
    __version__ = version("kvbiii_ml")
except PackageNotFoundError:
    pass

__all__ = [
    "__version__",
    "data_processing",
    "evaluation",
    "modeling",
]


def __getattr__(name: str):
    if name in __all__:
        return importlib.import_module(f"{__name__}.{name}")
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")
