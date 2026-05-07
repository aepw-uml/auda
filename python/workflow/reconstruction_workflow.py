from pathlib import Path
from typing import override

from common.dataset import Dataset, DatasetSchema
from common.experiment.persistence import (
    build_and_save_metric_table,
    save_hyperparameter_table,
    save_plots,
    save_time_table,
)
from common.workflow import Workflow
from experiment.reconstruction_task import (
    get_reconstruction_task,
)
from util.names import to_snake


class ReconstructionWorkflow(Workflow):
    """Runs reconstruction experiments and saves their artifacts."""

    @override
    def run(self, dataset: Dataset, schema: DatasetSchema, **context) -> None:
        """Runs the reconstruction workflow.

        Args:
            dataset: Dataset containing the feature matrix and target vector.
            schema: Schema describing the dataset columns and units.
            **context: Shared task and experiment configuration.
        """

        task = get_reconstruction_task(context)
        task.run(dataset, schema)

        for experiment in task.experiments:
            experiment.get_metrics()

        location = to_snake(context.get('location', ''))
        dir_path = Path('results') / (
            'multiple_reconstruction'
            if not location
            else f'multiple_reconstruction_{location}'
        )
        build_and_save_metric_table(task, dir_path)
        save_hyperparameter_table(task, dir_path)
        save_plots(task, dir_path)
        save_time_table(task, dir_path)
