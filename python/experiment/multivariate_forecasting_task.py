from typing import override

import numpy as np
from common.dataset import Dataset, DatasetSchema
from common.metrics import RegressionMetrics, average_regression_metrics
from dataset.pw_drivers import PWDrivers
from experiment.forecasting_task import (
    ForecastingTask,
    get_gaussian_process_forecasting,
    get_support_vector_regression_forecasting,
)
from experiment.nn_forecasting_experiment import NNForecastingExperiment
from step.model.isolation_forest import isolation_forest


class MultivariateForecastingTask(ForecastingTask):
    """Runs forecasting experiments on multivariate feature matrices."""

    @override
    def check_dataset(self, dataset: Dataset) -> None:
        """Validates the dataset required by multivariate forecasting.

        Args:
            dataset: Dataset containing the feature matrix and target vector.

        Raises:
            ValueError: If the dataset is incompatible with forecasting.
        """

        X, y = dataset.X, dataset.y

        if y is None:
            raise ValueError('Label data is required for forecasting task.')

        if X.shape[0] != y.shape[0]:
            raise ValueError(
                'Number of samples in features and labels must be the same.'
            )

        if y.ndim != 1:
            raise ValueError('Labels must be one-dimensional.')


def get_nn_forecasting(
    masked_dataset: Dataset, **context
) -> NNForecastingExperiment:
    """Builds a neural-network forecasting experiment.

    Args:
        masked_dataset: Dataset containing the feature matrix and target vector
            for training the neural network. The dataset may have been masked
            to exclude certain locations or time periods.
        context: Additional context for configuring the experiment.

    Returns:
        The configured neural-network forecasting experiment.
    """

    experiment = NNForecastingExperiment(
        name='NN Forecasting',
        description=(
            'Forecast the target series with a neural network using all '
            'available input features.'
        ),
    )

    experiment.set_context(pretraining_dataset=masked_dataset, **context)

    return experiment


def get_pretraining_dataset(
    context: dict[str, str],
    seed: int,
) -> Dataset:
    """Builds the auxiliary dataset used for neural-network pretraining.

    Args:
        context: Workflow context containing the optional target location.
        seed: Seed used by isolation forest when filtering outliers.

    Returns:
        The filtered out-of-location dataset for NN pretraining.
    """

    location = context.get('location', '')
    masked_dataset, _ = PWDrivers().fetch('', location)
    X, y = masked_dataset.X, masked_dataset.y
    if y is None:
        raise ValueError('Target variable y is required for pretraining.')

    result = isolation_forest(X, y, contamination=0.2, seed=seed)
    return Dataset(X=result.X_inliers, y=result.y_inliers)


def get_multivariate_forecasting_task(
    context: dict[str, str],
) -> MultivariateForecastingTask:
    """Builds the multivariate forecasting task.

    Args:
        context: Shared context to inject into every forecasting experiment.

    Returns:
        The configured multivariate forecasting task.
    """

    seed = int(context.get('seed', 471))
    masked_dataset = get_pretraining_dataset(context, seed)

    task = MultivariateForecastingTask(name='MultivariateForecasting')
    task.set_context(**context)
    task.add(get_nn_forecasting(masked_dataset))
    task.add(get_gaussian_process_forecasting())
    task.add(get_support_vector_regression_forecasting())
    return task


def run_multivariate_forecasting_task(
    dataset: Dataset,
    schema: DatasetSchema,
    context: dict[str, str] | None = None,
) -> MultivariateForecastingTask:
    """Runs the multivariate forecasting task.

    Args:
        dataset: Dataset containing the feature matrix and target vector.
        schema: Schema describing the dataset columns and units.
        context: Optional shared context for the task.

    Returns:
        The multivariate forecasting task after all experiments finish.
    """

    if context is None:
        context = {}

    task = get_multivariate_forecasting_task(context)
    task.run(dataset, schema)
    return task


def run_multivariate_forecasting_tasks(
    num_experiments: int,
    dataset: Dataset,
    schema: DatasetSchema,
    context: dict[str, str] | None = None,
    seed: int = 471,
) -> tuple[list[MultivariateForecastingTask], dict[str, RegressionMetrics]]:
    """Runs multiple multivariate forecasting tasks with different seeds.

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

    tasks: list[MultivariateForecastingTask] = []
    for experiment_seed in experiment_seeds:
        run_context = {} if context is None else dict(context)
        run_context['seed'] = str(int(experiment_seed))
        task = run_multivariate_forecasting_task(dataset, schema, run_context)
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


__all__ = [
    'MultivariateForecastingTask',
    'get_gaussian_process_forecasting',
    'get_multivariate_forecasting_task',
    'get_nn_forecasting',
    'get_pretraining_dataset',
    'get_support_vector_regression_forecasting',
    'run_multivariate_forecasting_task',
    'run_multivariate_forecasting_tasks',
]
