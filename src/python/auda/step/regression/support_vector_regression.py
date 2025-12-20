from typing import override

import numpy as np
from auda.step import get_dataset_from_step
from auda.step.plot import PlotStep
from auda.step.plot.plot_curve import PlotCurve
from auda.step.plot.plot_scatter_plot import PlotScatterPlot
from auda.step.spec import Dataset, Spec
from auda.utils.pipeline import IOValueMap, Step, step
from sklearn.svm import SVR


@step(
    id='RG-SVR',
    description='Trains a Support Vector Regression (SVR) model for '
    'predictive analysis.',
    input_specs=[
        Spec.ON.optional(Spec.DATASET.name),
        Spec.C.optional(1.0),
        Spec.EPSILON.optional(0.1),
        Spec.KERNEL.optional('rbf'),
    ],
    output_specs=[
        Spec.MODEL,
        Spec.C,
        Spec.EPSILON,
        Spec.NUM_SUPPORT_VECTORS,
        Spec.DUAL_COEFFICIENTS,
        Spec.INTERCEPT,
        Spec.GAMMA,
    ],
)
class SupportVectorRegression(Step):
    @override
    def run(
        self, on: str | Dataset, c: float, epsilon: float, kernel: str
    ) -> IOValueMap:
        from sklearn.svm import SVR

        c = float(c)
        epsilon = float(epsilon)

        X, y = get_dataset_from_step(self, on)

        svr_model = SVR(kernel=kernel, C=c, epsilon=epsilon)
        svr_model.fit(X, y)
        num_support_vectors = len(svr_model.support_.tolist())
        dual_coefficients = (
            svr_model.dual_coef_.ravel().astype(float).tolist()  # type: ignore
        )
        intercept = float(svr_model.intercept_[0])
        gamma = svr_model.gamma

        return {
            Spec.MODEL.name: svr_model,
            Spec.C.name: c,
            Spec.EPSILON.name: epsilon,
            Spec.NUM_SUPPORT_VECTORS.name: num_support_vectors,
            Spec.DUAL_COEFFICIENTS.name: dual_coefficients,
            Spec.INTERCEPT.name: intercept,
            Spec.GAMMA.name: gamma,
        }


@step(
    id='PL-SVR',
    description='Generates plots for Support Vector Regression (SVR) model '
    'results.',
    input_specs=[
        Spec.ON.optional(Spec.DATASET.name).desc('Dataset to plot against.'),
        Spec.MODEL.desc('Trained SVR model'),
        Spec.X_MEAN,
        Spec.X_STD,
        Spec.Y_MEAN.optional(0.0),
        Spec.Y_STD.optional(1.0),
        Spec.DATASET_SCEHMA,
    ],
    output_specs=[Spec.FIGURE, Spec.AXES],
)
class SupportVectorRegressionPlotter(PlotStep):
    @override
    def run(
        self,
        on: str | Dataset,
        model: SVR,
        x_mean: np.ndarray,
        x_std: np.ndarray,
        y_mean: float,
        y_std: float,
    ) -> IOValueMap:
        SAMPLE_POINT_SIZE = 60
        ORIGINAL_SAMPLE_COLOR = 'cornflowerblue'
        SUPPORT_VECTOR_EDGE_COLOR = 'black'
        LINE_COLOR = 'orange'
        LINE_LABEL = 'SVR Fit'
        EPSILON_TUBE_COLOR = 'orange'
        EPSILON_TUBE_ALPHA = 0.25
        EPSILON_TUBE_LABEL = 'ε-tube'

        X_true, y_true = get_dataset_from_step(self, on)
        self.check_feature_dimension(X_true, expected_dimension=1)

        x_true = X_true.ravel()

        # ---- Create a plot
        figure, axes = self.create_plot_or_default()

        # ---- Scatter original samples
        self.single_dispatch(
            PlotScatterPlot,
            {
                **self._inputs,
                Spec.ON.name: on,
                Spec.FIGURE.name: figure,
                Spec.AXES.name: axes,
                Spec.SAMPLE_POINT_SIZE.name: SAMPLE_POINT_SIZE + 20,
                Spec.SAMPLE_POINT_COLOR.name: ORIGINAL_SAMPLE_COLOR,
                Spec.SAMPLE_POINT_LABEL.name: 'Original Samples',
            },
        )

        # ---- Scatter support vectors
        support_indices = model.support_.tolist()
        x_support = x_true[support_indices]
        y_support = y_true[support_indices]

        self.single_dispatch(
            PlotScatterPlot,
            {
                **self._inputs,
                Spec.ON.name: (x_support.reshape(-1, 1), y_support),
                Spec.FIGURE.name: figure,
                Spec.AXES.name: axes,
                Spec.SAMPLE_POINT_SIZE.name: SAMPLE_POINT_SIZE,
                Spec.SAMPLE_POINT_COLOR.name: ORIGINAL_SAMPLE_COLOR,
                Spec.SAMPLE_POINT_EDGE_COLOR.name: SUPPORT_VECTOR_EDGE_COLOR,
                Spec.SAMPLE_POINT_LABEL.name: 'Support Vectors',
            },
        )

        # ---- Plot the SVR curve
        x_min, x_max = x_true.min(), x_true.max()
        x_mean_scalar = x_mean[0]
        x_curve = np.linspace(x_min, x_max, 300).reshape(-1, 1)
        x_curve_std = (x_curve - x_mean_scalar) / x_std
        y_curve_std = model.predict(x_curve_std)
        x_curve = x_curve_std * x_std + x_mean_scalar
        y_curve = y_curve_std * y_std + y_mean

        self.single_dispatch(
            PlotCurve,
            {
                **self._inputs,
                Spec.ON.name: (x_curve.reshape(-1, 1), y_curve),
                Spec.FIGURE.name: figure,
                Spec.AXES.name: axes,
                Spec.LINE_COLOR.name: LINE_COLOR,
                Spec.LINE_LABEL.name: LINE_LABEL,
            },
        )

        # ---- Plot the SVR tube
        epsilon = model.epsilon
        epsilon = epsilon * y_std

        axes.fill_between(
            x_curve.ravel(),
            y_curve - epsilon,
            y_curve + epsilon,
            color=EPSILON_TUBE_COLOR,
            alpha=EPSILON_TUBE_ALPHA,
            label=EPSILON_TUBE_LABEL,
            zorder=2,
        )

        axes.grid(True, alpha=0.25)
        axes.legend()
        figure.tight_layout()

        return self.regular_output(figure, axes)
