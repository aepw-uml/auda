from pathlib import Path
from typing import override

from common.dataset import Dataset, DatasetSchema
from common.experiment.persistence import (
    collect_metrics_by_name,
    save_hyperparameter_table,
    save_metric_summary_plot,
    save_metric_table,
    save_plots,
    save_time_table,
    summarize_metrics_by_name,
)
from common.metrics import RegressionMetricName
from common.workflow import Workflow
from experiment.forecasting_task import run_forecasting_tasks
from util.names import to_snake


class MultipleForecastingWorkflow(Workflow):
    """Runs repeated univariate forecasting experiments."""

    @override
    def run(self, dataset: Dataset, schema: DatasetSchema, **context) -> None:
        """Runs repeated forecasting and saves aggregate artifacts.

        Args:
            dataset: Dataset containing the feature matrix and target vector.
            schema: Schema describing the dataset columns and units.
            **context: Shared task and experiment configuration.
        """

        num_experiments = int(context.get('num_experiments', '16'))
        seed = int(context.get('seed', '42'))
        tasks, _ = run_forecasting_tasks(
            num_experiments, dataset, schema, context, seed=seed
        )
        metrics_by_name = collect_metrics_by_name(tasks)
        average_metrics, std_metrics = summarize_metrics_by_name(
            metrics_by_name
        )

        location = to_snake(context.get('location', ''))
        workflow_name = context.get('workflow_name')
        dir_path = Path('results') / (
            workflow_name
            if workflow_name
            else (
                'multiple_forecasting'
                if not location
                else f'multiple_forecasting_{location}'
            )
        )
        save_metric_table(average_metrics, dir_path)

        plot_metric: RegressionMetricName = context.get(
            'plot_metric', 'wape'
        )

        representative_task = tasks[0]
        if num_experiments > 1:
            save_metric_summary_plot(metrics_by_name, dir_path, plot_metric)
            for experiment in representative_task.experiments:
                experiment.context['metric_summary_mean'] = average_metrics[
                    experiment.name
                ]
                experiment.context['metric_summary_std'] = std_metrics[
                    experiment.name
                ]
                experiment.context['plot_metric'] = plot_metric

        save_hyperparameter_table(representative_task, dir_path)
        save_plots(representative_task, dir_path)
        save_time_table(representative_task, dir_path)
