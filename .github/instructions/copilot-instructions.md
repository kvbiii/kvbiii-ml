---
applyTo: '**'
---

# Python Coding Guidelines

## Code Style
- Prioritize conciseness: use list/dict comprehensions, lambda functions, and built-in functions over verbose loops
- Optimize for performance and efficiency
- Never use `#` comments in code - use docstrings for documentation
- Use descriptive variable and function names (concise but clear)

## Documentation
- All functions must have Google-style docstrings with the following format:
- Brief description on the first line
- Empty line
- `Args:` section with type hints inline (type | type for unions)
- `Returns:` section (if applicable)
- `Raises:` section (if applicable)

## Type Hints
- Use modern Python type hints (e.g., `list[str]`, `dict[str, int]`)
- Use union operator `|` instead of `Union` (e.g., `str | None`)
- Always include type hints for function parameters and return values

## Examples

### Function Docstring Format
```python
def function_name(
    param1: type1,
    param2: type2 | None = None,
    optional_param: type3 = default_value
) -> return_type:
    """
    Brief description of what the function does.

    Args:
        param1 (type1): Description of param1.
        param2 (type2, optional): Description of param2. Defaults to None.
        optional_param (type3, optional): Description. Defaults to default_value.

    Returns:
        return_type: Description of return value.

    Raises:
        ExceptionType: When and why this exception is raised.
    """
```

### Code Conciseness
Prefer:
```python
result = [x * 2 for x in items if x > 0]
mapping = {k: v.upper() for k, v in data.items()}
```

Over:
```python
result = []
for x in items:
    if x > 0:
        result.append(x * 2)
```