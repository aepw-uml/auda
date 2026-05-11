from typing import cast, override

import numpy as np
from common.dataset import Dataset, DatasetSchema
from common.experiment.forecasting_experiment import ForecastingExperiment
from common.metrics import RegressionMetrics, average_regression_metrics
from common.task import Task
from step.evaluator.complexity_key import (
    gaussian_process_regression_complexity_key,
    ridge_regression_complexity_key,
    support_vector_regression_complexity_key,
)
from step.model.arima_regression import ARIMARegression
from step.model.drift_baseline import DriftBaseline
from step.model.exponential_smoothing import ExponentialSmoothing
from step.model.gaussian_process_regression import GaussianProcessRegression
from step.model.naive_persistence import NaivePersistence
from step.model.ridge_regression import RidgeRegression
from step.model.support_vector_regression import SupportVectorRegression
from step.model.theil_sen_regression import TheilSenRegression
from step.plot.arima_regression import ARIMARegressionPlotter
from step.plot.gaussian_process_regression import (
    GaussianProcessRegressionPlotter,
)
from step.plot.ridge_regression import RidgeRegressionPlotter
from step.plot.support_vector_regression import SupportVectorRegressionPlotter
from step.plot.theil_sen_regression import TheilSenRegressionPlotter


def get_naive_persistence_forecasting(**_) -> ForecastingExperiment:
    return ForecastingExperiment(
        name='Naive Persistence',
        description=(
            'Forecast the original time series with naive persistence.'
        ),
        regressor_cls=NaivePersistence,
    )


def get_drift_baseline_forecasting(**_) -> ForecastingExperiment:
    return ForecastingExperiment(
        name='Drift Baseline',
        description=('Forecast the original time series with drift baseline.'),
        regressor_cls=DriftBaseline,
    )


def get_exponential_smoothing_forecasting(**_) -> ForecastingExperiment:
    experiment = ForecastingExperiment(
        name='Exponential Smoothing',
        description=(
            'Forecast the original time series with exponential smoothing.'
        ),
        regressor_cls=ExponentialSmoothing,
    )
    experiment.set_context(use_scaler=False, use_target_scaler=False)
    return experiment


def get_arima_forecasting(**context) -> ForecastingExperiment:
    experiment = ForecastingExperiment(
        name='ARIMA Regression',
        description=(
            'Forecast the original time series with an auto-ARIMA model.'
        ),
        regressor_cls=ARIMARegression,
    )

    experiment.set_context(
        **context,
        use_scaler=False,
        use_target_scaler=False,
        plotter_factory=ARIMARegressionPlotter,
    )
    experiment.hyperparameters = {
        'replace': True,
        'auto': True,
        'max_p': 2,
        'max_d': 1,
        'max_q': 2,
        'max_order': 5,
        'information_criterion': 'aic',
        'trend': 'n',
    }

    return experiment


def get_theil_sen_forecasting(**_) -> ForecastingExperiment:
    experiment = ForecastingExperiment(
        name='Theil-Sen Regression',
        description=(
            'Forecast the original time series with robust Theil-Sen '
            'regression.'
        ),
        regressor_cls=TheilSenRegression,
    )

    experiment.set_context(
        plotter_factory=TheilSenRegressionPlotter,
    )
    experiment.hyperparameters = {
        'replace': True,
        'window_size': 7,
    }

    return experiment


def get_ridge_regression_forecasting(**_) -> ForecastingExperiment:
    experiment = ForecastingExperiment(
        name='Ridge Regression',
        description=(
            'Forecast the original time series with ridge regression.'
        ),
        regressor_cls=RidgeRegression,
    )

    experiment.set_context(
        plotter_factory=RidgeRegressionPlotter,
        tuning_parameters={
            'hyperparameter_names': ['degree', 'alpha'],
            'search_space': [[(2.0, 9.0)], [(1e-6, 1e3)]],
            'sampling_scales': ['uniform', 'log_uniform'],
            'complexity_key': ridge_regression_complexity_key,
        },
    )

    return experiment


def get_gaussian_process_forecasting(**_) -> ForecastingExperiment:
    experiment = ForecastingExperiment(
        name='Gaussian Process Regression',
        description=(
            'Forecast the original time series with Gaussian process '
            'regression.'
        ),
        regressor_cls=GaussianProcessRegression,
    )

    experiment.set_context(
        plotter_factory=GaussianProcessRegressionPlotter,
        tuning_parameters={
            'hyperparameter_names': ['length_scale', 'noise_level'],
            'search_space': [[(1e-3, 1e3)], [(1e-6, 1e1)]],
            'sampling_scales': ['log_uniform', 'log_uniform'],
            'complexity_key': gaussian_process_regression_complexity_key,
        },
    )

    return experiment


def get_support_vector_regression_forecasting(
    **context: str,
) -> ForecastingExperiment:
    tune_gamma = context.get('svr_tune_gamma', '0') == '1'
    experiment = ForecastingExperiment(
        name='Support Vector Regression'
        + (' (with gamma tuning)' if tune_gamma else ''),
        description=(
            'Forecast the original time series with support vector regression.'
        ),
        regressor_cls=SupportVectorRegression,
    )

    if tune_gamma:
        tuning_parameters = {
            'hyperparameter_names': ['C', 'epsilon', 'gamma'],
            'search_space': [
                [(0.1, 100.0)],
                [(0.001, 1.0)],
                [(0.001, 1.0)],
            ],
            'sampling_scales': [
                'log_uniform',
                'log_uniform',
                'log_uniform',
            ],
            'complexity_key': support_vector_regression_complexity_key,
        }
    else:
        tuning_parameters = {
            'hyperparameter_names': ['C', 'epsilon'],
            'search_space': [
                [(0.1, 100.0)],
                [(0.001, 1.0)],
            ],
            'sampling_scales': [
                'log_uniform',
                'log_uniform',
            ],
            'complexity_key': support_vector_regression_complexity_key,
        }

    experiment.set_context(
        plotter_factory=SupportVectorRegressionPlotter,
        tuning_parameters=tuning_parameters,
        **context,
    )

    return experiment


class ForecastingTask(Task):
    @override
    def run(self, dataset: Dataset, schema: DatasetSchema) -> None:
        """Executes the forecasting task by running all configured forecasting
        experiments.
        """

        super().run(dataset)

        X, y = dataset.X, dataset.y
        assert y is not None

        seed = int(self.context.get('seed', 42))
        self.logger.info(f'Using seed {seed}.')

        metric = self.context.get('metric', 'wape')
        self.logger.info(f'Using metric "{metric}" for hyperparameter tuning.')

        for experiment in self.experiments:
            experiment = cast(ForecastingExperiment, experiment)

            # Set the random seed for reproducibility.
            experiment.seed = seed

            # Inject schema into the experiment context.
            experiment.context['schema'] = schema

            # Inject metric into the experiment context for hyperparameter
            # tuning if tuning parameters are defined.
            if 'tuning_parameters' in experiment.context:
                experiment.context['tuning_parameters']['metric'] = metric

            experiment.setup(X, y, **self.context)
            experiment.run()
            experiment.logger.info(experiment.get_metrics())

    @override
    def check_dataset(self, dataset: Dataset) -> None:
        X, y = dataset.X, dataset.y

        if y is None:
            raise ValueError('Label data is required for forecasting task.')

        if X.shape[0] != y.shape[0]:
            raise ValueError(
                'Number of samples in features and labels must be the same.'
            )

        if X.shape[1] != 1 or y.ndim != 1:
            raise ValueError(
                'Features must have exactly one column and labels must be '
                'one-dimensional.'
            )


def get_forecasting_task(context: dict[str, str]) -> ForecastingTask:
    """Builds a forecasting task with all standard forecasting experiments.

    Args:
        context: Shared context to inject into every forecasting experiment.

    Returns:
        The configured forecasting task.
    """

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
    return task


def run_forecasting_task(
    dataset: Dataset,
    schema: DatasetSchema,
    context: dict[str, str] | None = None,
) -> ForecastingTask:
    """Runs the standard forecasting task.

    Args:
        dataset: Dataset containing the feature matrix and target vector.
        schema: Schema describing the dataset columns and units.
        context: Optional shared context for the task.

    Returns:
        The forecasting task after all experiments finish.
    """

    if context is None:
        context = {}

    task = get_forecasting_task(context)
    task.run(dataset, schema)
    return task


def run_forecasting_tasks(
    num_experiments: int,
    dataset: Dataset,
    schema: DatasetSchema,
    context: dict[str, str] | None = None,
    seed: int = 42,
) -> tuple[list[ForecastingTask], dict[str, RegressionMetrics]]:
    """Runs multiple forecasting tasks with different seeds.

    Args:
        num_experiments: Number of tasks to run.
        dataset: Dataset containing the feature matrix and target vector.
        schema: Schema describing the dataset columns and units.
        context: Optional shared context for every task.
        seed: Seed used to sample the per-run task seeds.

    Returns:
        A tuple containing the tasks and their averaged metrics by experiment
        name.
    """

    rng = np.random.default_rng(seed)
    experiment_seeds = rng.integers(0, 2 << 31, size=num_experiments)

    tasks: list[ForecastingTask] = []
    for experiment_seed in experiment_seeds:
        run_context = {} if context is None else dict(context)
        run_context['seed'] = str(int(experiment_seed))
        task = run_forecasting_task(dataset, schema, run_context)
        tasks.append(task)

    metrics_lists_by_name: dict[str, list[RegressionMetrics]] = {}
    for task in tasks:
        for experiment in task.experiments:
            metrics_lists_by_name.setdefault(experiment.name, []).append(
                experiment.get_metrics()
            )

    metrics_by_name: dict[str, RegressionMetrics] = {}
    for name, metrics_list in metrics_lists_by_name.items():
        metrics_by_name[name] = average_regression_metrics(metrics_list)

    return tasks, metrics_by_name
