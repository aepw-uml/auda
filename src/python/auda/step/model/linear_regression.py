from typing import override

import numpy as np
from auda.step.dataset import DatasetBasedStep
from auda.step.model import ModelBasedStep
from auda.step.plot import PlotStep
from auda.step.spec import Dataset, Spec
from auda.utils.pipeline import IOValueMap, step


@step(
    id='MD-LR',
    description='Trains a linear regression model to fit nonlinear trends.',
    input_specs=[Spec.ON.optional(Spec.DATASET.name)],
    output_specs=[Spec.MODEL, Spec.COEFFICIENTS, Spec.INTERCEPT],
)
class LinearRegressionModel(DatasetBasedStep):
    @override
    def run(self, on: str | Dataset) -> IOValueMap:
        from sklearn.linear_model import LinearRegression

        X, y = self.get_dataset_from_on(on)

        model = LinearRegression()
        model.fit(X, y)

        return {
            Spec.MODEL.name: model,
            Spec.COEFFICIENTS.name: np.array(model.coef_),
            Spec.INTERCEPT.name: float(model.intercept_),
        }


@step(
    id='PL-LR-3D',
    description='Plots the results of the linear regression model.',
    input_specs=[
        Spec.ON.optional(Spec.DATASET.name),
        Spec.MODEL,
        Spec.COEFFICIENTS,
        Spec.INTERCEPT,
        Spec.X_MEAN.optional(),
        Spec.X_STD.optional(),
        Spec.Y_MEAN.optional(),
        Spec.Y_STD.optional(),
    ],
    output_specs=[Spec.FIGURE, Spec.AXES],
)
class PlotLinearRegression(DatasetBasedStep, ModelBasedStep, PlotStep):
    @override
    def run(
        self,
        model,
        on: str | Dataset,
        coefficients: np.ndarray,
        intercept: float,
        x_mean: np.ndarray | None,
        x_std: np.ndarray | None,
        y_mean: float | None,
        y_std: float | None,
    ) -> IOValueMap:
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D  # noqa
        from sklearn.linear_model import LinearRegression

        self.verify_model(model, LinearRegression)

        X, y = self.get_dataset_from_on(on)
        self.check_feature_dimension(X, 2)

        figure = plt.figure(figsize=(8, 6))
        axes = figure.add_subplot(111, projection='3d')

        # ---- Scatter original data points
        axes.scatter(
            X[:, 0],
            X[:, 1],
            y,  # type: ignore
            color='blue',
            s=50,
            edgecolor='white',
            label='Samples',
        )

        return self.regular_output(figure, axes)
