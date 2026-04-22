from pathlib import Path
from typing import override

from common.dataset import Dataset, DatasetSchema
from common.experiment.persistence import save_metric_table
from common.workflow import Workflow
from experiment.reconstruction_task import (
    run_reconstruction_tasks,
)


class MultipleReconstructionWorkflow(Workflow):
    @override
    def run(self, dataset: Dataset, schema: DatasetSchema, **context) -> None:
        super().run(**context)

        num_experiments = int(context.get('num_experiments', '16'))
        seed = int(context.get('seed', '42'))
        _, average_metrics = run_reconstruction_tasks(
            num_experiments, dataset, schema, context, seed=seed
        )

        dir_path = Path('results') / 'multiple_reconstruction'
        save_metric_table(average_metrics, dir_path)
