## Style Guide

### Python Style Guide

In this project, the Python codebase should comply with the [Google Python Style Guide](https://google.github.io/styleguide/pyguide.html). Since this is a data analysis project, all types, classes, methods, and functions must be documented with proper docstrings. In particular, class methods and functions in the `python/step` directory should include docstrings that clearly describe their specific functionality.

There should be a blank line between the docstring of a function or method and the first line of code inside it. One-line docstring summaries should be written in the third-person singular form. If a function does not return a value, its return type should be `None` in the docstring. If the returned value type is `None`, the dcostring should not include a "Returns" section.

For example:

```python
def average(num: list[float]) -> float:
    """Averages the numbers in the list and return the result.

    Args:
        num: A list of numbers to be averaged.

    Returns:
        The average of the numbers in the list.
    """

    return sum(num) / len(num)

def print_average(num: list[float]) -> None:
    """Prints the average of the numbers in the list.

    Args:
        num: A list of numbers to be averaged and printed.
    """

    avg = average(num)
    print(f"The average is: {avg}")
```

All methods that override methods from parent classes should be marked with `@override` to indicate that the method is overriding a method from a parent class. These methods do not have to have a docstring unless some additional information is needed to clarify the method's functionality.
