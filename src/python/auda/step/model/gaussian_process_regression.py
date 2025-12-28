from typing import override

from auda.step.dataset import DatasetBasedStep
from auda.step.spec import Dataset, Spec
from auda.utils.pipeline import IOValueMap, step


@step(
    id='MD-GPR',
    description='Gaussian Process Regression (GPR) Model Step',
    input_specs=[
        Spec.ON.optional(Spec.DATASET.name),
        Spec.LENGTH_SCALE.optional(1.0),
        Spec.NOISE_LEVEL.optional(1e-2),
    ],
    output_specs=[
        Spec.MODEL,
        Spec.LENGTH_SCALE,
        Spec.NOISE_LEVEL,
    ],
)
class GaussianProcessRegressionModel(DatasetBasedStep):
    @override
    def run(
        self,
        on: str | Dataset,
        length_scale: float,
        noise_level: float,
    ) -> IOValueMap:
        from sklearn.gaussian_process import GaussianProcessRegressor
        from sklearn.gaussian_process.kernels import RBF, WhiteKernel

        X, y = self.get_dataset_from_on(on)

        # Kernel = smooth trend (RBF) + observation noise
        kernel = RBF(
            length_scale=float(length_scale), length_scale_bounds=(1e-10, 1e3)
        ) + WhiteKernel(
            noise_level=float(noise_level), noise_level_bounds=(1e-6, 1e1)
        )

        model = GaussianProcessRegressor(
            kernel=kernel,
            random_state=42,
        )
        model.fit(X, y)

        return {
            Spec.MODEL.name: model,
            Spec.LENGTH_SCALE.name: float(length_scale),
            Spec.NOISE_LEVEL.name: float(noise_level),
        }
