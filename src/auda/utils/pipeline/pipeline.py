from typing import Any, Callable, Dict, List

from ._task import Task, TaskSpec


class Pipeline:
    def __init__(self, task_specs: List[TaskSpec]):
        """
        Initializes a Pipeline instance.

        Attributes:
            task_specs: A list of TaskSpec instances representing the tasks in the
                pipeline.
        """
        self._task_specs: List[TaskSpec] = task_specs
        self.callbacks: List[Callable[[], None]] = []

    def run(self, initial_inputs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Runs the pipeline by executing each task in sequence.

        Args:
            initial_inputs: A dictionary of initial inputs to the pipeline.

        Returns:
            A dictionary containing the final outputs after all tasks have been
            executed.
        """
        tasks: List[Task] = [spec.instantiate(self) for spec in self._task_specs]

        context: Dict[str, Any] = dict(initial_inputs)
        for task in tasks:
            task.set_inputs(context)
            task.run()

            # Merge both this task’s inputs and outputs into context
            context.update(task.get_outputs())

        return context

    def schedule(self, callback: Callable[[], None]) -> None:
        """
        Schedules a callback to be executed after the pipeline run.

        Args:
            callback: A callable function to be executed after the pipeline run.
        """
        self.callbacks.append(callback)
