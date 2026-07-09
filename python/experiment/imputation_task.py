from pathlib import Path
from typing import cast, override

import numpy as np
from common.dataset import Dataset, DatasetSchema
from common.experiment.persistence import (
    build_and_save_metric_table,
    save_hyperparameter_table,
    save_plots,
    save_time_table,
)
from common.experiment.imputation_experiment import (
    ImputationExperiment,
)
from common.metrics import RegressionMetrics, average_regression_metrics
from common.task import Task
from step.evaluator.complexity_key import (
    gaussian_process_regression_complexity_key,
    ridge_regression_complexity_key,
    support_vector_regression_complexity_key,
)
from step.model.cubic_spline import CubicSpline
from step.model.gaussian_process_regression import GaussianProcessRegression
from step.model.linear_interpolation import LinearInterpolation
from step.model.moving_average_interpolation import MovingAverageInterpolation
from step.model.ridge_regression import RidgeRegression
from step.model.support_vector_regression import SupportVectorRegression
from step.model.theil_sen_regression import TheilSenRegression
from step.plot.gaussian_process_regression import (
    GaussianProcessRegressionPlotter,
)
from step.plot.ridge_regression import RidgeRegressionPlotter
from step.plot.support_vector_regression import SupportVectorRegressionPlotter
from step.plot.theil_sen_regression import TheilSenRegressionPlotter


def get_linear_interpolation_imputation(
    **_,
) -> ImputationExperiment:
    """Builds a linear-interpolation imputation experiment."""

    return ImputationExperiment(
        name='Linear Interpolation',
        description=(
            'Impute the original time series with linear interpolation.'
        ),
        regressor_cls=LinearInterpolation,
    )


def get_moving_average_interpolation_imputation(
    **_,
) -> ImputationExperiment:
    """Builds a moving-average-interpolation imputation experiment."""

    return ImputationExperiment(
        name='Moving Average Interpolation',
        description=(
            'Impute the original time series with moving average '
            'interpolation.'
        ),
        regressor_cls=MovingAverageInterpolation,
    )


def get_cubic_spline_interpolation_imputation(
    **_,
) -> ImputationExperiment:
    """Builds a cubic-spline-interpolation imputation experiment."""

    return ImputationExperiment(
        name='Cubic Spline Interpolation',
        description=(
            'Impute the original time series with cubic spline '
            'interpolation.'
        ),
        regressor_cls=CubicSpline,
    )


def get_theil_sen_imputation(**_) -> ImputationExperiment:
    """Builds a Theil-Sen imputation experiment."""

    experiment = ImputationExperiment(
        name='Theil-Sen Regression',
        description=(
            'Impute the original time series with robust Theil-Sen '
            'regression.'
        ),
        regressor_cls=TheilSenRegression,
    )
    experiment.set_context(plotter_factory=TheilSenRegressionPlotter)
    experiment.hyperparameters = {
        'replace': True,
        'window_size': 7,
    }
    return experiment


def get_ridge_regression_imputation(
    **_,
) -> ImputationExperiment:
    """Builds a ridge-regression imputation experiment."""

    experiment = ImputationExperiment(
        name='Ridge Regression',
        description=(
            'Impute the original time series with ridge regression.'
        ),
        regressor_cls=RidgeRegression,
    )

    experiment.set_context(
        plotter_factory=RidgeRegressionPlotter,
        tuning_parameters={
            'search_type': 'grid',
            'hyperparameter_names': ['degree', 'alpha'],
            'search_space': [[(2.0, 9.0)], [(1e-6, 1e3)]],
            'sampling_scales': ['uniform', 'log_uniform'],
            'complexity_key': ridge_regression_complexity_key,
        },
    )

    return experiment


def get_gaussian_process_imputation(
    **_,
) -> ImputationExperiment:
    """Builds a Gaussian-process imputation experiment."""

    experiment = ImputationExperiment(
        name='Gaussian Process Regression',
        description=(
            'Impute the original time series with Gaussian process '
            'regression.'
        ),
        regressor_cls=GaussianProcessRegression,
    )

    experiment.set_context(
        plotter_factory=GaussianProcessRegressionPlotter,
        tuning_parameters={
            'search_type': 'grid',
            'hyperparameter_names': ['length_scale', 'noise_level'],
            'search_space': [[(1e-3, 1e3)], [(1e-6, 1e1)]],
            'sampling_scales': ['log_uniform', 'log_uniform'],
            'complexity_key': gaussian_process_regression_complexity_key,
        },
    )

    return experiment


def get_support_vector_regression_imputation(
    **context: str,
) -> ImputationExperiment:
    """Builds a support-vector-regression imputation experiment."""

    experiment = ImputationExperiment(
        name='Support Vector Regression',
        description=(
            'Impute the original time series with support vector '
            'regression.'
        ),
        regressor_cls=SupportVectorRegression,
    )

    if context.get('svr_tune_gamma', '0') == '1':
        tuning_parameters = {
            'search_type': 'grid',
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
            'search_type': 'grid',
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


class ImputationTask(Task):
    """Runs the configured imputation experiments for one task."""

    @override
    def run(self, dataset: Dataset, schema: DatasetSchema) -> None:
        """Runs the experiments after injecting shared context.

        Args:
            dataset: Dataset containing the feature matrix and target vector.
            schema: Schema describing the dataset columns and units.
        """

        super().run(dataset)

        X, y = dataset.X, dataset.y
        assert y is not None

        seed = int(self.context.get('seed', 42))
        self.logger.info(f'Using seed {seed}.')

        metric = self.context.get('metric', 'wape')
        self.logger.info(f'Using metric "{metric}" for hyperparameter tuning.')

        for experiment in self.experiments:
            experiment = cast(ImputationExperiment, experiment)
            experiment.seed = seed
            experiment.context['schema'] = schema

            if 'tuning_parameters' in experiment.context:
                experiment.context['tuning_parameters']['metric'] = metric

            experiment.setup(X, y, **self.context)
            experiment.run()
            experiment.logger.info(experiment.get_metrics())

    @override
    def check_dataset(self, dataset: Dataset) -> None:
        """Validates the dataset required by imputation experiments.

        Args:
            dataset: Dataset containing the feature matrix and target vector.

        Raises:
            ValueError: If the dataset is incompatible with imputation.
        """

        X, y = dataset.X, dataset.y

        if y is None:
            raise ValueError(
                'Label data is required for imputation experiments.'
            )

        if X.shape[0] != y.shape[0]:
            raise ValueError(
                'Number of samples in features and labels must be the same.'
            )

        if y.ndim != 1:
            raise ValueError('Labels must be one-dimensional.')


def get_imputation_task(
    context: dict[str, str],
) -> ImputationTask:
    """Builds the imputation task.

    Args:
        context: Shared context to inject into every imputation experiment.

    Returns:
        The configured imputation task.
    """

    task = ImputationTask(name='Imputation')
    task.set_context(**context)
    task.add(get_linear_interpolation_imputation(**context))
    task.add(get_moving_average_interpolation_imputation(**context))
    task.add(get_cubic_spline_interpolation_imputation(**context))
    task.add(get_theil_sen_imputation(**context))
    task.add(get_ridge_regression_imputation(**context))
    task.add(get_gaussian_process_imputation(**context))
    task.add(get_support_vector_regression_imputation(**context))
    return task


def run_imputation_task(
    dataset: Dataset,
    schema: DatasetSchema,
    context: dict[str, str] | None = None,
) -> ImputationTask:
    """Runs the imputation task.

    Args:
        dataset: Dataset containing the feature matrix and target vector.
        schema: Schema describing the dataset columns and units.
        context: Optional shared context for the task.

    Returns:
        The imputation task after all experiments finish.
    """

    if context is None:
        context = {}

    task = get_imputation_task(context)
    task.run(dataset, schema)
    return task


def save_imputation_task_results(
    task: ImputationTask,
) -> None:
    """Saves imputation metrics, hyperparameters, plots, and timings."""

    task_path = Path('results') / 'imputation'
    build_and_save_metric_table(task, task_path)
    save_hyperparameter_table(task, task_path)
    save_plots(task, task_path)
    save_time_table(task, task_path)


def run_imputation_tasks(
    num_experiments: int,
    dataset: Dataset,
    schema: DatasetSchema,
    context: dict[str, str] | None = None,
    seed: int = 42,
) -> tuple[list[ImputationTask], dict[str, RegressionMetrics]]:
    """Runs multiple imputation tasks with different seeds.

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

    tasks: list[ImputationTask] = []
    for experiment_seed in experiment_seeds:
        run_context = {} if context is None else dict(context)
        run_context['seed'] = str(int(experiment_seed))
        task = run_imputation_task(dataset, schema, run_context)
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
