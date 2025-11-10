from typing import Any, Dict, List, Tuple

from auda.utils.pipeline import Pipeline, TaskSpec, get_all_specs


def get_task_spec(task_spec_id: str) -> TaskSpec:
    """
    Retrieves a TaskSpec by its ID.

    Args:
        task_spec_id: The ID of the TaskSpec to retrieve.
    """

    for task_spec in get_all_specs():
        if task_spec.id == task_spec_id:
            return task_spec

    raise ValueError(f'Task spec with id {task_spec_id} not found.')


def run_pipeline(
    task_spec_id_list: List[str], inputs: Dict[str, Any] | None = None
) -> Tuple[Dict[str, Any], Pipeline]:
    """
    Runs a pipeline of tasks specified by their IDs.

    Args:
        task_spec_id_list: A list of task spec IDs to include in the pipeline.
        inputs: A dictionary of inputs to the pipeline.

    Returns:
        A tuple containing the outputs of the pipeline and the Pipeline object.
    """
    task_specs = [get_task_spec(task_id) for task_id in task_spec_id_list]
    pipeline = Pipeline(task_specs)

    return pipeline.run(inputs or {}), pipeline
