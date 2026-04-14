from pathlib import Path
from typing import override

from common.dataset import Dataset, DatasetSchema
from common.experiment.persistence import (
    save_hyperparameter_table,
    save_metric_table,
    save_plots,
    save_time_table,
)
from common.task import Task
from experiment.reconstruction_experiments import (
    get_reconstruction_experiment_group,
)


class ReconstructionTask(Task):
    """Runs reconstruction experiments and saves their artifacts."""

    @override
    def run(self, dataset: Dataset, schema: DatasetSchema, **context) -> None:
        """Runs the reconstruction task.

        Args:
            dataset: Dataset containing the feature matrix and target vector.
            schema: Schema describing the dataset columns and units.
            **context: Shared task and experiment configuration.
        """

        group = get_reconstruction_experiment_group(context)
        group.run(dataset, schema)

        for experiment in group.experiments:
            experiment.get_metrics()

        task_path = Path('results') / 'reconstruction'
        save_metric_table(group, task_path)
        save_hyperparameter_table(group, task_path)
        save_plots(group, task_path)
        save_time_table(group, task_path)
