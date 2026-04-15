from pathlib import Path
from typing import override

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
    getARIMAForecasting,
    getDriftBaselineForecasting,
    getExponentialSmoothingForecasting,
    getGaussianProcessForecasting,
    getNaivePersistenceForecasting,
    getRidgeRegressionForecasting,
    getSupportVectorRegressionForecasting,
    getTheilSenForecasting,
)


class ForecastingTask(Task):
    @override
    def run(self, dataset: Dataset, schema: DatasetSchema, **context) -> None:
        group = ForecastingExperimentGroup(name='Forecasting Experiments')
        group.set_context(**context)
        group.add(getNaivePersistenceForecasting(**context))
        group.add(getDriftBaselineForecasting(**context))
        group.add(getExponentialSmoothingForecasting(**context))
        group.add(getRidgeRegressionForecasting(**context))
        group.add(getGaussianProcessForecasting(**context))
        group.add(getSupportVectorRegressionForecasting(**context))
        group.add(getTheilSenForecasting(**context))
        group.add(getARIMAForecasting(**context))

        group.run(dataset, schema)

        for experiment in group.experiments:
            experiment.get_metrics()

        # Task path to save the results of the forecasting experiments.
        task_path = Path('results') / 'forecasting'

        # Save the metric table for the forecasting experiments.
        build_and_save_metric_table(group, task_path)

        # Save a hyperparameter table and save it to a file.
        save_hyperparameter_table(group, task_path)

        # Save the plots of each experiment.
        save_plots(group, task_path)

        # Save time.
        save_time_table(group, task_path)
