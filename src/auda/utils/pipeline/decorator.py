from typing import Any, Dict

from ._task import TaskSpec
from .registry import register


def task(
    *,
    id: str,
    kind: str,
    description: str,
    input_specs: Dict[str, Any] | None = None,
    output_specs: Dict[str, Any] | None = None,
):
    """
    Decorate a Task subclass to register a TaskSpec for it.
    """
    input_specs = input_specs or {}
    output_specs = output_specs or {}

    def _wrap(cls):
        spec = TaskSpec(
            id=id,
            kind=kind,
            description=description,
            implementation=cls,
            input_specs=input_specs,
            output_specs=output_specs,
        )
        register(spec)
        return cls

    return _wrap
