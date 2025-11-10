from typing import override

from auda.dat.datasets import DatasetOSName, UnlabeledSamples
from auda.dat.models import ModelOSName
from auda.utils.pipeline import IOSpec, Task, task

from .__common import MODEL_KIND, ModelISName


@task(
    id='MD-CM',
    kind=MODEL_KIND,
    description='Computes the correlation matrix among input features.',
    input_specs={
        ModelISName.SAMPLES: IOSpec(dtype=UnlabeledSamples),
    },
    output_specs={
        ModelOSName.CORRELATION_MATRIX: IOSpec(dtype=UnlabeledSamples),
    },
)
class CorrelationModel(Task):
    @override
    def run(self) -> None:
        """
        Compute the correlation matrix of the dataset features.
        """
        import numpy as np

        samples: UnlabeledSamples = self.get_input(DatasetOSName.SAMPLES)
        X = np.array(samples)
        n = X.shape[0]

        # ---- Centering the data
        X = X - np.mean(X, axis=0)

        # ---- Calculate the covariance matrix
        covariance_matrix: np.ndarray = (X.T @ X) / (n - 1)

        # ---- Calculate the correlation matrix
        std_dev = np.sqrt(np.diag(covariance_matrix))
        D_inv = np.diag(1 / std_dev)
        correlation_matrix = D_inv @ covariance_matrix @ D_inv

        # ---- Populate outputs
        self.set_output(ModelOSName.CORRELATION_MATRIX, correlation_matrix)
