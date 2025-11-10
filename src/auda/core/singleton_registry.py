from typing import Any, Dict, Protocol, Type, TypeVar, cast


class NoArgConstructible(Protocol):
    def __init__(self) -> None: ...


T = TypeVar('T', bound=NoArgConstructible)


class SingletonRegistry:
    def __init__(self) -> None:
        self._singletons: Dict[Type[Any], Any] = {}

    def get(self, cls: Type[T]) -> T:
        """
        Return the singleton instance for `cls`, creating it if needed.
        Type-safe: if you pass `Type[Foo]`, you'll get `Foo`.
        """
        if cls not in self._singletons:
            self._singletons[cls] = cls()

        return cast(T, self._singletons[cls])
