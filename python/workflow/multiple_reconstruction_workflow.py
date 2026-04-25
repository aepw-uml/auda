from pathlib import Path
from typing import override

from common.dataset import Dataset, DatasetSchema
from common.experiment.persistence import save_metric_table
from common.workflow import Workflow
from experiment.reconstruction_task import (
    run_reconstruction_tasks,
)
from util.names import to_kebab


class MultipleReconstructionWorkflow(Workflow):
    @override
    def run(self, dataset: Dataset, schema: DatasetSchema, **context) -> None:
        super().run(**context)

        num_experiments = int(context.get('num_experiments', '16'))
        seed = int(context.get('seed', '42'))
        _, average_metrics = run_reconstruction_tasks(
            num_experiments, dataset, schema, context, seed=seed
        )

        location = to_kebab(context.get('location', ''))
        dir_path = Path('results') / (
            'multiple_reconstruction'
            if not location
            else f'multiple_reconstruction_{location}'
        )
        save_metric_table(average_metrics, dir_path)
