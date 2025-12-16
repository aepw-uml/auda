from __future__ import annotations

from types import NoneType, UnionType
from typing import (
    Annotated,
    Any,
    ForwardRef,
    Literal,
    Union,
    get_args,
    get_origin,
)


def _name(t: Any) -> str:
    if t is Any:
        return 'Any'
    if t in (None, NoneType):
        return 'None'
    return getattr(t, '__name__', str(t))


def format_type(t: Any) -> str:
    """Formats a type annotation (incl. PEP 604 unions like X | Y) into a
    readable string.
    """
    # Forward refs like "MyClass" in quotes
    if isinstance(t, ForwardRef):
        return t.__forward_arg__

    origin = get_origin(t)
    args = get_args(t)

    # Bare (non-parameterized) types
    if origin is None:
        return _name(t)

    # Union (supports both typing.Union[...] and X | Y)
    if origin in (Union, UnionType):
        # Optional[T] sugar for Union[T, None]
        if NoneType in args and len(args) == 2:
            non_none = next(a for a in args if a is not NoneType)
            return f'Optional[{format_type(non_none)}]'
        # Keep order as given
        return ' | '.join(format_type(a) for a in args)

    # Literal
    if origin is Literal:
        # Use repr for literal values
        return f'Literal[{", ".join(repr(a) for a in args)}]'

    # Annotated[T, ...]
    if origin is Annotated and args:
        base, *meta = args
        meta_str = ', '.join(repr(m) for m in meta)
        return f'Annotated[{format_type(base)}, {meta_str}]'

    # Tuple[T, ...]
    if origin is tuple and args:
        if len(args) == 2 and args[1] is Ellipsis:
            return f'tuple[{format_type(args[0])}, ...]'
        return f'Tuple[{", ".join(format_type(a) for a in args)}]'

    # Generic containers / parametrized classes (list, dict, set, etc.)
    if args:
        name = _name(origin).title()
        return f'{name}[{", ".join(format_type(a) for a in args)}]'

    # Fallback
    return _name(origin)
