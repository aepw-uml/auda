from pathlib import Path
from typing import override

from common.dataset import Dataset, DatasetSchema
from common.experiment.persistence import (
    collect_metrics_by_name,
    save_metric_summary_plot,
    save_metric_table,
    save_plots,
    summarize_metrics_by_name,
)
from common.metrics import RegressionMetricName
from common.workflow import Workflow
from experiment.imputation_task import (
    run_imputation_tasks,
)
from util.names import to_snake


class MultipleImputationWorkflow(Workflow):
    @override
    def run(self, dataset: Dataset, schema: DatasetSchema, **context) -> None:
        super().run(**context)

        num_experiments = int(context.get('num_experiments', '16'))
        seed = int(context.get('seed', '42'))
        tasks, _ = run_imputation_tasks(
            num_experiments, dataset, schema, context, seed=seed
        )
        metrics_by_name = collect_metrics_by_name(tasks)
        average_metrics, std_metrics = summarize_metrics_by_name(
            metrics_by_name
        )

        location = to_snake(context.get('location', ''))
        dir_path = Path('results') / (
            'multiple_imputation'
            if not location
            else f'multiple_imputation_{location}'
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

        save_plots(representative_task, dir_path)
