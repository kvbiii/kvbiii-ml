# Copilot Code Review Instructions

Apply these guidelines when reviewing changes to Python files (`*.py`) in this repository.

## Type Hints

Flag any public function, method, or `__init__` signature missing type hints. Use modern syntax: `str | None` (not `Optional[str]`), `list[str]` (not `List[str]`), `dict[str, int]` (not `Dict[str, int]`).

## Imports

Imports must be grouped into three sections — standard library, third-party, local/workspace — separated by a blank line, alphabetically sorted within each group. Flag wildcard imports (`from module import *`) and imports placed inside functions or conditionals.

## Exception Handling

Flag bare `except:` or silent `except Exception: pass`. Exceptions should be caught with a specific type and re-raised with context: `raise NewError(...) from exc`. Never swallow exceptions silently.

## Docstrings

Every public function, method, and class needs a Google-style docstring (Args/Returns/Raises where relevant). Private (`_`-prefixed) functions need at least a one-line docstring if non-trivial. Do not flag missing module-level docstrings — this repo intentionally omits them.

## Scope

Do not flag formatting (quote style, line length, trailing commas) or naming conventions — those are already enforced by `black` and `pylint` in CI. Focus review comments on correctness, the rules above, and algorithmic complexity (e.g., O(n²) patterns where a set/dict lookup would suffice).
