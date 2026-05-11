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
from experiment.forecasting_task import run_forecasting_task


class ForecastingWorkflow(Workflow):
    @override
    def run(self, dataset: Dataset, schema: DatasetSchema, **context) -> None:
        tune_search_type = context.get('tune_search_type', 'random')
        workflow_name = context.get('workflow_name', 'forecasting')
        dir_path = (
            Path('results') / f'{workflow_name}_{tune_search_type}_search'
        )

        task = run_forecasting_task(dataset, schema, context)

        for experiment in task.experiments:
            experiment.get_metrics()

        build_and_save_metric_table(task, dir_path)
        save_hyperparameter_table(task, dir_path)
        save_plots(task, dir_path)
        save_time_table(task, dir_path)
