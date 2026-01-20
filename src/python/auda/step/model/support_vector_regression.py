from typing import List, override

import numpy as np
from auda.step.dataset import DatasetBasedStep
from auda.step.model import ModelBasedStep
from auda.step.plot import PlotStep
from auda.step.plot.plot_curve import PlotCurve
from auda.step.plot.plot_scatter_plot import PlotScatterPlot
from auda.step.plot.plot_set import PlotSet
from auda.step.spec import Dataset, Spec
from auda.utils.pipeline import IOValueMap, step


@step(
    id='MD-SVR',
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
class SupportVectorRegression(DatasetBasedStep):
    @override
    def run(
        self, on: str | Dataset, c: float, epsilon: float, kernel: str
    ) -> IOValueMap:
        from sklearn.svm import SVR

        c = float(c)
        epsilon = float(epsilon)

        X, y = self.get_dataset_from_on(on)

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
    id='PD-SVR',
    description='Generates predictions using a trained Support Vector '
    'Regression model.',
    input_specs=[
        Spec.MODEL,
        Spec.X_PRED_VALUES,
        Spec.X_MEAN,
        Spec.X_STD,
        Spec.Y_MEAN.optional(0.0),
        Spec.Y_STD.optional(1.0),
    ],
    output_specs=[Spec.PRED_DATASET],
)
class SupportVectorRegressionPredictor(ModelBasedStep):
    @override
    def run(
        self,
        model,
        x_pred_values: List[float],
        x_mean: np.ndarray,
        x_std: np.ndarray,
        y_mean: float,
        y_std: float,
    ) -> IOValueMap:
        from sklearn.svm import SVR

        self.verify_model(model, SVR)

        # ---- Standardize prediction inputs
        x_pred = np.array(x_pred_values, dtype=float).reshape(-1, 1)
        x_pred_std = (x_pred - x_mean) / x_std

        # ---- Generate predictions and reverse standardization
        y_pred_std = model.predict(x_pred_std.reshape(-1, 1))
        y_pred = y_pred_std * y_std + y_mean

        return {
            Spec.PRED_DATASET.name: (x_pred, y_pred),
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
        Spec.DATASET_SCHEMA,
        Spec.PRED_DATASET.optional(),
        Spec.CURVE_EXTEND_MARGIN_RATIO.optional(0.05),
        Spec.TEST_SET.optional(),
    ],
    output_specs=[Spec.FIGURE, Spec.AXES],
)
class SupportVectorRegressionPlotter(
    PlotStep, DatasetBasedStep, ModelBasedStep
):
    @override
    def run(
        self,
        on: str | Dataset,
        model,
        x_mean: np.ndarray,
        x_std: np.ndarray,
        y_mean: float,
        y_std: float,
        pred_dataset: Dataset | None,
        curve_extend_margin_ratio: float,
        test_set: Dataset | None,
    ) -> IOValueMap:
        from matplotlib.pyplot import axvspan
        from sklearn.svm import SVR

        SAMPLE_POINT_SIZE = 60
        ORIGINAL_SAMPLE_COLOR = 'cornflowerblue'
        PREDICTED_SAMPLE_COLOR = 'green'
        SUPPORT_VECTOR_EDGE_COLOR = 'black'
        LINE_COLOR = 'orange'
        LINE_LABEL = 'SVR Fit'
        EPSILON_TUBE_COLOR = 'orange'
        EPSILON_TUBE_ALPHA = 0.25
        EPSILON_TUBE_LABEL = 'ε-tube'

        self.verify_model(model, SVR)

        X_true, y_true = self.get_dataset_from_on(on)
        self.check_feature_dimension(X_true, expected_dimension=1)

        x_true = X_true.ravel()

        # ---- Create x_all for later use
        x_all = x_true
        if pred_dataset is not None:
            x_pred, _ = pred_dataset
            x_all = np.concatenate([x_true, x_pred.ravel()])
        if test_set is not None:
            x_test, _ = test_set
            x_all = np.concatenate([x_all, x_test.ravel()])

        # ---- Create a plot
        figure, axes = self.create_plot_or_default()

        # ---- Plot original data points (training set)
        self.single_dispatch(
            PlotScatterPlot,
            {
                **self._inputs,
                Spec.ON.name: on,
                Spec.FIGURE.name: figure,
                Spec.AXES.name: axes,
                Spec.SAMPLE_POINT_SIZE.name: SAMPLE_POINT_SIZE,
                Spec.SAMPLE_POINT_COLOR.name: ORIGINAL_SAMPLE_COLOR,
                Spec.SAMPLE_POINT_LABEL.name: 'Original Samples',
            },
        )

        # ---- Plot support vectors
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

        # ---- Plot test set data points (if given)
        if test_set is not None:
            x_test, y_test = test_set
            self.single_dispatch(
                PlotScatterPlot,
                {
                    **self._inputs,
                    Spec.ON.name: (x_test, y_test),
                    Spec.FIGURE.name: figure,
                    Spec.AXES.name: axes,
                    Spec.SAMPLE_POINT_SIZE.name: SAMPLE_POINT_SIZE,
                    Spec.SAMPLE_POINT_COLOR.name: 'green',
                    Spec.SAMPLE_POINT_LABEL.name: 'Test Samples',
                },
            )

        # ---- Plot predicted data points (if given)
        if pred_dataset is not None:
            x_pred, y_pred = pred_dataset
            self.single_dispatch(
                PlotScatterPlot,
                {
                    **self._inputs,
                    Spec.ON.name: (x_pred, y_pred),
                    Spec.FIGURE.name: figure,
                    Spec.AXES.name: axes,
                    Spec.SAMPLE_POINT_SIZE.name: SAMPLE_POINT_SIZE,
                    Spec.SAMPLE_POINT_COLOR.name: PREDICTED_SAMPLE_COLOR,
                    Spec.SAMPLE_POINT_LABEL.name: 'Predictions',
                },
            )

        # ---- Plot the SVR curve
        x_min, x_max = x_all.min(), x_all.max()
        x_min, x_max = self.extend_range(
            x_min, x_max, curve_extend_margin_ratio
        )

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

        # ---- Highlight Forecast Region
        if pred_dataset is not None:
            last_train_year = float(x_true.max())

            axvspan(
                last_train_year,
                axes.get_xlim()[1],
                color='gray',
                alpha=0.15,
                label='Forecast Region',
            )

        self.single_dispatch(
            PlotSet,
            {
                Spec.GRID_ALPHA.name: 0.25,
                Spec.LEGEND_LOCATION.name: 'best',
                **self._inputs,
                Spec.FIGURE.name: figure,
                Spec.AXES.name: axes,
            },
        )

        return self.regular_output(figure, axes)
