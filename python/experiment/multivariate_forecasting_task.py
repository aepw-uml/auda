from typing import override

from common.dataset import Dataset
from experiment.forecasting_task import (
    ForecastingTask,
    get_gaussian_process_forecasting,
    get_support_vector_regression_forecasting,
)
from experiment.nn_forecasting_experiment import NNForecastingExperiment


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


__all__ = [
    'MultivariateForecastingTask',
    'get_gaussian_process_forecasting',
    'get_nn_forecasting',
    'get_support_vector_regression_forecasting',
]
