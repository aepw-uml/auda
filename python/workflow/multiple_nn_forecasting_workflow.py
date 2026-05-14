from pathlib import Path
from typing import cast, override

from common.dataset import Dataset, DatasetSchema
from common.experiment.persistence import (
    collect_metrics_by_name,
    save_metric_summary_plot,
    save_metric_table,
    summarize_metrics_by_name,
)
from common.metrics import RegressionMetricName
from common.task.task import Task
from common.workflow import Workflow
from util.names import to_snake
from workflow.nn_forecasting_workflow import run_nn_forecasting_tasks


class MultipleNNForecastingWorkflow(Workflow):
    """Runs repeated neural-network forecasting experiments."""

    @override
    def run(self, dataset: Dataset, schema: DatasetSchema, **context) -> None:
        """Runs repeated NN forecasting and saves average metrics.

        Args:
            dataset: Dataset containing the feature matrix and target vector.
            schema: Schema describing the dataset columns and units.
            **context: Shared task and experiment configuration.
        """

        num_experiments = int(context.get('num_experiments', '16'))
        seed = int(context.get('seed', '471'))
        tasks, _ = run_nn_forecasting_tasks(
            num_experiments, dataset, schema, context, seed=seed
        )
        metrics_by_name = collect_metrics_by_name(cast(list[Task], tasks))
        average_metrics, _ = summarize_metrics_by_name(metrics_by_name)

        location = to_snake(context.get('location', ''))
        workflow_name = context.get('workflow_name')
        dir_path = Path('results') / (
            workflow_name
            if workflow_name
            else (
                'multiple_nn_forecasting'
                if not location
                else f'multiple_nn_forecasting_{location}'
            )
        )
        save_metric_table(average_metrics, dir_path)

        plot_metric: RegressionMetricName = context.get('plot_metric', 'wape')
        if num_experiments > 1:
            save_metric_summary_plot(metrics_by_name, dir_path, plot_metric)
