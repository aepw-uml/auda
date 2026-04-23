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
        tune_search_type = context.get('tune_search_type', 'random')
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

        dir_path = Path('results') / f'forecasting_{tune_search_type}_search'
        build_and_save_metric_table(task, dir_path)
        save_hyperparameter_table(task, dir_path)
        save_plots(task, dir_path)
        save_time_table(task, dir_path)
