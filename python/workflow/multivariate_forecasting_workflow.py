from pathlib import Path
from typing import override

from common.dataset import Dataset, DatasetSchema
from common.experiment.persistence import (
    build_and_save_metric_table,
)
from common.workflow import Workflow
from experiment.multivariate_forecasting_task import (
    MultivariateForecastingTask,
    get_gaussian_process_forecasting,
    get_nn_forecasting,
    get_support_vector_regression_forecasting,
)


class MultivariateForecastingWorkflow(Workflow):
    @override
    def run(self, dataset: Dataset, schema: DatasetSchema, **context) -> None:
        task = MultivariateForecastingTask(name='MultivariateForecasting')
        task.set_context(**context)

        task.add(get_nn_forecasting())
        task.add(get_support_vector_regression_forecasting())
        task.add(get_gaussian_process_forecasting())

        task.run(dataset, schema)

        for experiment in task.experiments:
            experiment.get_metrics()

        workflow_name: str = context.get(
            'workflow_name', 'multivariate_forecasting'
        )
        dir_path = Path('results') / workflow_name
        build_and_save_metric_table(task, dir_path)
