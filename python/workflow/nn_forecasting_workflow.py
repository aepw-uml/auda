from pathlib import Path
from typing import override

import numpy as np
from common.dataset import Dataset, DatasetSchema
from common.experiment.persistence import (
    build_and_save_metric_table,
    save_hyperparameter_table,
    save_time_table,
)
from common.metrics import RegressionMetrics, average_regression_metrics
from common.workflow import Workflow
from experiment.nn_forecasting_experiment import (
    NNForecastingExperiment,
    NNForecastingTask,
)


def _format_contamination_rate(contamination: float) -> str:
    """Formats an anomaly contamination rate for context and table labels.

    Args:
        contamination: The anomaly contamination rate to format.

    Returns:
        A compact string representation of the contamination rate.
    """

    return f'{contamination:g}'


def _parse_anomaly_contamination_rates(
    context: dict[str, str],
) -> list[float]:
    """Parses anomaly contamination rates from workflow context.

    Args:
        context: Workflow context that may contain ``anomaly_contamination`` as
            a single value or comma-separated list.

    Returns:
        The parsed contamination rates, or an empty list when the option is not
        set.

    Raises:
        ValueError: If the option contains an empty or non-numeric value.
    """

    raw_value = context.get('anomaly_contamination')
    if raw_value is None:
        return []

    raw_rates = [rate.strip() for rate in raw_value.split(',')]
    if any(rate == '' for rate in raw_rates):
        raise ValueError(
            'Invalid anomaly_contamination value. Expected a comma-separated '
            'list of numbers, such as "0.1,0.2,0.3".'
        )

    try:
        contamination_rates = [float(rate) for rate in raw_rates]
    except ValueError as error:
        raise ValueError(
            'Invalid anomaly_contamination value. Expected a comma-separated '
            'list of numbers, such as "0.1,0.2,0.3".'
        ) from error

    invalid_rates = [
        contamination
        for contamination in contamination_rates
        if contamination <= 0.0 or contamination > 0.5
    ]
    if invalid_rates:
        raise ValueError(
            'Invalid anomaly_contamination value. Contamination rates must be '
            'greater than 0 and less than or equal to 0.5.'
        )

    return contamination_rates


class NNForecastingWorkflow(Workflow):
    @override
    def run(self, dataset: Dataset, schema: DatasetSchema, **context) -> None:
        task = self.run_task(dataset, schema, context)

        for experiment in task.experiments:
            print(experiment.get_metrics())

        dir_path = Path('results') / 'nn_forecasting'
        build_and_save_metric_table(task, dir_path)
        save_hyperparameter_table(task, dir_path)
        save_time_table(task, dir_path)

    def run_task(
        self,
        dataset: Dataset,
        schema: DatasetSchema,
        context: dict[str, str],
    ) -> NNForecastingTask:
        """Runs one neural-network forecasting task.

        Args:
            dataset: Dataset containing the feature matrix and target vector.
            schema: Schema describing the dataset columns and units.
            context: Shared task context.

        Returns:
            The completed neural-network forecasting task.
        """

        contamination_rates = _parse_anomaly_contamination_rates(context)
        task_context = dict(context)
        if contamination_rates:
            task_context.setdefault('use_isolation_forest', '1')

        if len(contamination_rates) > 1:
            task_context.pop('anomaly_contamination', None)

        task = NNForecastingTask('NN Forecasting')
        task.set_context(**task_context)

        if not contamination_rates:
            task.add(
                NNForecastingExperiment(
                    'NN Forecasting',
                    'Forecast the plastic waste generation using a neural '
                    'network model.',
                )
            )
        else:
            for contamination in contamination_rates:
                contamination_label = _format_contamination_rate(
                    contamination
                )
                experiment_name = 'NN Forecasting'
                if len(contamination_rates) > 1:
                    experiment_name = (
                        f'{experiment_name} '
                        f'(contamination={contamination_label})'
                    )

                experiment = NNForecastingExperiment(
                    experiment_name,
                    'Forecast the plastic waste generation using a neural '
                    'network model.',
                )
                experiment.set_context(
                    anomaly_contamination=contamination_label
                )
                task.add(experiment)

        task.run(dataset, schema)
        return task


def run_nn_forecasting_tasks(
    num_experiments: int,
    dataset: Dataset,
    schema: DatasetSchema,
    context: dict[str, str] | None = None,
    seed: int = 471,
) -> tuple[list[NNForecastingTask], dict[str, RegressionMetrics]]:
    """Runs multiple neural-network forecasting tasks with different seeds.

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

    workflow = NNForecastingWorkflow()
    tasks: list[NNForecastingTask] = []
    for experiment_seed in experiment_seeds:
        run_context = {} if context is None else dict(context)
        run_context['seed'] = str(int(experiment_seed))
        task = workflow.run_task(dataset, schema, run_context)
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
