from typing import override

from common.dataset import Dataset, DatasetSchema
from common.task import Task
from experiment.nn_forecasting_experiment import (
    NNForecastingExperiment,
    NNForecastingExperimentGroup,
)


class NNForecastingTask(Task):
    @override
    def run(self, dataset: Dataset, schema: DatasetSchema, **context) -> None:
        group = NNForecastingExperimentGroup('NN Forecasting')
        group.set_context(**context)
        group.add(
            NNForecastingExperiment(
                'NN Forecasting',
                'Forecast the plastic waste generation using a neural network '
                'model.',
            )
        )

        group.run(dataset, schema)

        for experiment in group.experiments:
            print(experiment.get_metrics())
