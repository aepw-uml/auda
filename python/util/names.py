def to_kebab(name: str) -> str:
    """Convert a name to kebab-case.

    Args:
        name: The name to convert.

    Example:
        >>> to_kebab('MyClass')
        'myclass'
        >>> to_kebab('my_class')
        'my-class'
        >>> to_kebab('my class')
        'my-class'
    """

    return name.lower().replace('_', '-').replace(' ', '-')
