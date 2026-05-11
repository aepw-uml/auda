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

        task = NNForecastingTask('NN Forecasting')
        task.set_context(**context)
        task.add(
            NNForecastingExperiment(
                'NN Forecasting',
                'Forecast the plastic waste generation using a neural network '
                'model.',
            )
        )

        task.run(dataset, schema)
        return task
