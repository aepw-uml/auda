from typing import override

import numpy as np
from auda.step.dataset import DatasetBasedStep, DatasetSchema
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
        Spec.DATASET_SCHEMA,
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
        dataset_schema: DatasetSchema,
    ) -> IOValueMap:
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D  # noqa
        from sklearn.linear_model import LinearRegression

        self.verify_model(model, LinearRegression)

        X, y = self.get_dataset_from_on(on)
        self.check_feature_dimension(X, 2)

        figure = plt.figure(figsize=(8, 6))
        axes = figure.add_subplot(111, projection='3d')
        axes.set_box_aspect((1.7, 1.7, 1.7))

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

        # ---- Plot the linear plane
        # Build a grid in ORIGINAL feature space (so it overlays the scatter)
        x0_min, x0_max = float(X[:, 0].min()), float(X[:, 0].max())
        x1_min, x1_max = float(X[:, 1].min()), float(X[:, 1].max())

        # Add a small margin so points aren't on the boundary
        x0_min, x0_max = self.extend_range(x0_min, x0_max, margin_ratio=0.05)
        x1_min, x1_max = self.extend_range(x1_min, x1_max, margin_ratio=0.05)

        n0, n1 = 50, 50
        x0_lin = np.linspace(x0_min, x0_max, n0)
        x1_lin = np.linspace(x1_min, x1_max, n1)
        X0g, X1g = np.meshgrid(x0_lin, x1_lin)

        grid = np.column_stack([X0g.ravel(), X1g.ravel()])  # (n0*n1, 2)

        # If the model was trained on standardized X, standardize the grid too.
        # (Your pipeline typically standardizes using ST-BASIC / ZNorm.)
        if x_mean is not None and x_std is not None:
            grid_eval = (grid - x_mean.reshape(1, -1)) / x_std.reshape(1, -1)
        else:
            grid_eval = grid

        # Compute plane in the space the model coefficients correspond to
        # y_hat = intercept + w1*x1 + w2*x2
        w = np.asarray(coefficients, dtype=float).reshape(-1)
        if w.shape[0] != 2:
            raise ValueError(
                f'Expected 2 coefficients for 2D LR; got {w.shape[0]}.'
            )

        y_hat = float(intercept) + grid_eval @ w  # (n0*n1,)

        # If the model was trained on standardized y, reverse it for plotting
        if y_mean is not None and y_std is not None:
            y_hat = y_hat * float(y_std) + float(y_mean)

        Yg = y_hat.reshape(X0g.shape)

        # Draw the plane
        axes.plot_surface(
            X0g,
            X1g,
            Yg,
            alpha=0.35,
            linewidth=0,
            antialiased=True,
        )

        # Optional: labels/grid/legend polish
        z_label = (
            dataset_schema.label_names[0] if dataset_schema.label_names else ''
        )
        axes.set_xlabel(dataset_schema.feature_names[0])
        axes.set_ylabel(dataset_schema.feature_names[1])
        axes.set_zlabel(z_label)

        try:
            axes.legend(loc='best')
        except Exception:
            pass

        return self.regular_output(figure, axes)
