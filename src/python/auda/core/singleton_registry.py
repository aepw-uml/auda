from typing import Any, Dict, Protocol, Type, TypeVar, cast


class NoArgConstructible(Protocol):
    def __init__(self) -> None: ...


T = TypeVar('T', bound=NoArgConstructible)


class SingletonRegistry:
    def __init__(self) -> None:
        """Initializes the SingletonRegistry.

        Attributes:
            __singletons: A dictionary mapping classes to their singleton
            instances.
        """

        self.__singletons: Dict[Type[Any], Any] = {}

    def get(self, cls: Type[T]) -> T:
        """Returns the singleton instance for `cls`, creating it if needed."""

        if cls not in self.__singletons:
            self.__singletons[cls] = cls()

        return cast(T, self.__singletons[cls])
