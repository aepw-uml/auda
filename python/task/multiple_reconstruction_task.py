from pathlib import Path
from typing import override

from common.dataset import Dataset, DatasetSchema
from common.experiment.persistence import save_metric_table
from common.task import Task
from experiment.reconstruction_experiments import (
    run_reconstruction_experiment_groups,
)


class MultipleReconstructionTask(Task):
    @override
    def run(self, dataset: Dataset, schema: DatasetSchema, **context) -> None:
        num_experiments = int(context.get('num_experiments', '16'))
        seed = int(context.get('seed', '42'))
        _, average_metrics = run_reconstruction_experiment_groups(
            num_experiments, dataset, schema, context, seed=seed
        )

        task_path = Path('results') / 'multiple_reconstruction'
        save_metric_table(average_metrics, task_path)
