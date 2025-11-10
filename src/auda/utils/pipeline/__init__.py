from ._task import IOSpec, Task, TaskSpec
from .decorator import task
from .pipeline import Pipeline
from .registry import get_all_specs, register

__all__ = [
    'IOSpec',
    'Task',
    'TaskSpec',
    'Pipeline',
    'get_all_specs',
    'register',
    'task',
]
