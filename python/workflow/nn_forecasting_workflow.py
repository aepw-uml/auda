from pathlib import Path
from typing import override

from common.dataset import Dataset, DatasetSchema
from common.experiment.persistence import (
    build_and_save_metric_table,
    save_hyperparameter_table,
    save_time_table,
)
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
