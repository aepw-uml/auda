from pathlib import Path
from typing import override

from common.dataset import Dataset, DatasetSchema
from common.experiment.persistence import (
    build_and_save_metric_table,
)
from common.workflow import Workflow
from experiment.multivariate_forecasting_task import (
    run_multivariate_forecasting_task,
)


class MultivariateForecastingWorkflow(Workflow):
    """Runs multivariate forecasting experiments and saves their metrics."""

    @override
    def run(self, dataset: Dataset, schema: DatasetSchema, **context) -> None:
        """Runs the multivariate forecasting workflow.

        Args:
            dataset: Dataset containing the feature matrix and target vector.
            schema: Schema describing the dataset columns and units.
            **context: Shared task and experiment configuration.
        """

        task = run_multivariate_forecasting_task(dataset, schema, context)

        for experiment in task.experiments:
            experiment.get_metrics()

        workflow_name: str = context.get(
            'workflow_name', 'multivariate_forecasting'
        )
        dir_path = Path('results') / workflow_name
        build_and_save_metric_table(task, dir_path)
