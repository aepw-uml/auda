from typing import override

from common.dataset import Dataset, DatasetSchema
from common.workflow import Workflow
from experiment.nn_forecasting_experiment import (
    NNForecastingExperiment,
    NNForecastingTask,
)


class NNForecastingWorkflow(Workflow):
    @override
    def run(self, dataset: Dataset, schema: DatasetSchema, **context) -> None:
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

        for experiment in task.experiments:
            print(experiment.get_metrics())
