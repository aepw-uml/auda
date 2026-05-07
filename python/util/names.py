def to_snake(name: str) -> str:
    """Convert a name to snake_case.

    Args:
        name: The name to convert.

    Example:
        >>> to_snake('MyClass')
        'my_class'
        >>> to_snake('my-class')
        'my_class'
        >>> to_snake('my class')
        'my_class'
    """

    return name.lower().replace('-', '_').replace(' ', '_')
