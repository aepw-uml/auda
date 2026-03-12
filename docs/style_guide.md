## Style Guide

### Python Style Guide

In this project, the Python codebase should comply with the [Google Python Style Guide](https://google.github.io/styleguide/pyguide.html). Since this is a data analysis project, all types, classes, methods, and functions must be documented with proper docstrings. In particular, class methods and functions in the `python/step` directory should include docstrings that clearly describe their specific functionality.

There should be a blank line between the docstring of a function or method and the first line of code inside it. One-line docstring summaries should be written in the third-person singular form. If a function does not return a value, its return type should be `None` in the docstring.

For example:

```python
def average(num: list[num]) -> None:
    """Averages the numbers in the list and return the result.

    Args:
        num: A list of numbers to be averaged.

    Returns:
        The average of the numbers in the list.
    """

    return sum(num) / len(num)
```
