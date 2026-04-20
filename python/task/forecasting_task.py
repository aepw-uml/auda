from pathlib import Path
from typing import Literal, override

from common.dataset import Dataset, DatasetSchema
from common.experiment.persistence import (
    build_and_save_metric_table,
    save_hyperparameter_table,
    save_plots,
    save_time_table,
)
from common.task import Task
from experiment.forecasting_experiments import (
    ForecastingExperimentGroup,
    get_arima_forecasting,
    get_drift_baseline_forecasting,
    get_exponential_smoothing_forecasting,
    get_gaussian_process_forecasting,
    get_naive_persistence_forecasting,
    get_ridge_regression_forecasting,
    get_support_vector_regression_forecasting,
    get_theil_sen_forecasting,
)


class ForecastingTask(Task):
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

        group = ForecastingExperimentGroup(name='Forecasting Experiments')
        group.set_context(**context)
        group.add(get_naive_persistence_forecasting(**context))
        group.add(get_drift_baseline_forecasting(**context))
        group.add(get_exponential_smoothing_forecasting(**context))
        group.add(get_ridge_regression_forecasting(**context))
        group.add(get_gaussian_process_forecasting(**context))
        group.add(get_support_vector_regression_forecasting(**context))
        group.add(get_theil_sen_forecasting(**context))
        group.add(get_arima_forecasting(**context))

        group.run(dataset, schema)

        for experiment in group.experiments:
            experiment.get_metrics()

        # Task path to save the results of the forecasting experiments.
        task_path = Path('results') / f'forecasting_{tune_search_type}_search'

        # Save the metric table for the forecasting experiments.
        build_and_save_metric_table(group, task_path)

        # Save a hyperparameter table and save it to a file.
        save_hyperparameter_table(group, task_path)

        # Save the plots of each experiment.
        save_plots(group, task_path)

        # Save time.
        save_time_table(group, task_path)
