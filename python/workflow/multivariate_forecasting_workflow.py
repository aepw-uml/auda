from pathlib import Path
from typing import override

from common.dataset import Dataset, DatasetSchema
from common.experiment.persistence import (
    build_and_save_metric_table,
)
from common.workflow import Workflow
from dataset.pw_drivers import PWDrivers
from experiment.multivariate_forecasting_task import (
    MultivariateForecastingTask,
    get_gaussian_process_forecasting,
    get_nn_forecasting,
    get_support_vector_regression_forecasting,
)
from step.model.isolation_forest import isolation_forest


class MultivariateForecastingWorkflow(Workflow):
    @override
    def run(self, dataset: Dataset, schema: DatasetSchema, **context) -> None:
        seed = int(context.get('seed', 471))

        task = MultivariateForecastingTask(name='MultivariateForecasting')
        task.set_context(**context)

        # Build an auxiliary dataset for NN pretraining by excluding the target
        # location and removing outliers.
        location = context.get('location', '')
        masked_dataset, _ = PWDrivers().fetch('', location)
        X, y = masked_dataset.X, masked_dataset.y
        assert y is not None, 'Target variable y is None'
        result = isolation_forest(X, y, contamination=0.2, seed=seed)
        X, y = result.X_inliers, result.y_inliers
        masked_dataset = Dataset(X=X, y=y)

        # The NN forecasting experiment first pretrains on the auxiliary
        # out-of-location dataset, then fine-tunes on the requested location.
        task.add(get_nn_forecasting(masked_dataset))
        task.add(get_gaussian_process_forecasting())
        task.add(get_support_vector_regression_forecasting())

        task.run(dataset, schema)

        for experiment in task.experiments:
            experiment.get_metrics()

        workflow_name: str = context.get(
            'workflow_name', 'multivariate_forecasting'
        )
        dir_path = Path('results') / workflow_name
        build_and_save_metric_table(task, dir_path)
