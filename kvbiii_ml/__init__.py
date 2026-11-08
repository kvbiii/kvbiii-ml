from __future__ import annotations

import contextlib
import importlib
from importlib.metadata import PackageNotFoundError, version
from types import ModuleType

from . import data_processing, evaluation, modeling

__version__ = "0+unknown"
with contextlib.suppress(PackageNotFoundError):
    __version__ = version("kvbiii_ml")

__all__ = [
    "__version__",
    "data_processing",
    "evaluation",
    "modeling",
]


def __getattr__(name: str) -> ModuleType:
    if name in __all__:
        return importlib.import_module(f"{__name__}.{name}")
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")
