from pathlib import Path
from typing import Literal, override

from common.dataset import Dataset, DatasetSchema
from common.experiment.persistence import (
    build_and_save_metric_table,
    save_hyperparameter_table,
    save_plots,
    save_time_table,
)
from common.workflow import Workflow
from experiment.forecasting_task import (
    ForecastingTask,
    get_arima_forecasting,
    get_drift_baseline_forecasting,
    get_exponential_smoothing_forecasting,
    get_gaussian_process_forecasting,
    get_naive_persistence_forecasting,
    get_ridge_regression_forecasting,
    get_support_vector_regression_forecasting,
    get_theil_sen_forecasting,
)


class ForecastingWorkflow(Workflow):
    @override
    def run(self, dataset: Dataset, schema: DatasetSchema, **context) -> None:
        self._run(dataset, schema, tune_search_type='grid', **context)
        self._run(dataset, schema, tune_search_type='random', **context)

    def _run(
        self,
        dataset: Dataset,
        schema: DatasetSchema,
        tune_search_type: Literal['grid', 'random'],
        **context,
    ) -> None:
        context['tune_search_type'] = tune_search_type

        svr_gamma_context = context.copy()
        svr_gamma_context['svr_tune_gamma'] = '1'

        task = ForecastingTask(name='Forecasting')
        task.set_context(**context)
        task.add(get_naive_persistence_forecasting(**context))
        task.add(get_drift_baseline_forecasting(**context))
        task.add(get_exponential_smoothing_forecasting(**context))
        task.add(get_ridge_regression_forecasting(**context))
        task.add(get_gaussian_process_forecasting(**context))
        task.add(get_support_vector_regression_forecasting(**context))
        task.add(get_support_vector_regression_forecasting(**svr_gamma_context))
        task.add(get_theil_sen_forecasting(**context))
        task.add(get_arima_forecasting(**context))

        task.run(dataset, schema)

        for experiment in task.experiments:
            experiment.get_metrics()

        task_path = Path('results') / f'forecasting_{tune_search_type}_search'
        build_and_save_metric_table(task, task_path)
        save_hyperparameter_table(task, task_path)
        save_plots(task, task_path)
        save_time_table(task, task_path)
