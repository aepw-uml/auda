from typing import override

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
    id='MD-RR',
    description='Ridge Regression Model Step',
    input_specs=[
        Spec.ON.optional(Spec.DATASET.name),
        Spec.ALPHA.optional(1.0),
    ],
    output_specs=[
        Spec.MODEL,
        Spec.ALPHA,
        Spec.COEFFICIENTS,
        Spec.INTERCEPT,
    ],
)
class RidgeRegressionModel(DatasetBasedStep):
    @override
    def run(self, on: str | Dataset, alpha: float) -> IOValueMap:
        from sklearn.linear_model import Ridge

        X, y = self.get_dataset_from_on(on)

        model = Ridge(alpha=alpha)
        model.fit(X, y)

        return {
            Spec.MODEL.name: model,
            Spec.ALPHA.name: alpha,
            Spec.COEFFICIENTS.name: model.coef_,
            Spec.INTERCEPT.name: model.intercept_,
        }


@step(
    id='PL-RR',
    description='Generates plots for Ridge Regression model results.',
    input_specs=[
        Spec.ON.optional(Spec.DATASET.name).desc('Dataset to plot against.'),
        Spec.MODEL.desc('Trained ridge regression model'),
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
class RidgeRegressionPlotter(PlotStep, DatasetBasedStep, ModelBasedStep):
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
        from sklearn.linear_model import Ridge

        SAMPLE_POINT_SIZE = 60
        ORIGINAL_SAMPLE_COLOR = 'cornflowerblue'
        PREDICTED_SAMPLE_COLOR = 'green'
        LINE_COLOR = 'orange'
        LINE_LABEL = 'Ridge Fit'

        self.verify_model(model, Ridge)

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

        # ---- Plot the ridge regression curve
        x_min, x_max = x_all.min(), x_all.max()
        x_min, x_max = self.extend_range(
            x_min, x_max, curve_extend_margin_ratio
        )

        x_mean_scalar = float(x_mean[0])
        x_std_scalar = float(x_std[0])
        x_curve = np.linspace(x_min, x_max, 300).reshape(-1, 1)
        x_curve_std = (x_curve - x_mean_scalar) / x_std_scalar
        y_curve_std = model.predict(x_curve_std)
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
