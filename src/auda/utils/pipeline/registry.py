from __future__ import annotations

from typing import Dict, List

from ._task import TaskSpec

_task_specs: Dict[str, TaskSpec] = {}


def register(spec: TaskSpec) -> None:
    """
    Register a TaskSpec.

    Args:
        spec: The TaskSpec to register.
    """
    if spec.id in _task_specs:
        raise ValueError(f'Duplicate TaskSpec id: {spec.id}')

    _task_specs[spec.id] = spec


def get_all_specs() -> List[TaskSpec]:
    return list(_task_specs.values())
