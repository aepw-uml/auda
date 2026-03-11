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
    id='MD-PRR',
    description='Polynomial Ridge Regression Model Step',
    input_specs=[
        Spec.ON.optional(Spec.DATASET.name),
        Spec.DEGREE.optional(3),
        Spec.ALPHA.optional(1.0),
    ],
    output_specs=[
        Spec.MODEL,
        Spec.DEGREE,
        Spec.ALPHA,
        Spec.COEFFICIENTS,
        Spec.INTERCEPT,
    ],
)
class PolynomialRidgeRegressionModel(DatasetBasedStep):
    @override
    def run(self, on: str | Dataset, degree: int, alpha: float) -> IOValueMap:
        from sklearn.linear_model import Ridge
        from sklearn.pipeline import Pipeline as SkPipeline
        from sklearn.preprocessing import PolynomialFeatures

        X, y = self.get_dataset_from_on(on)
        degree = int(degree)

        model = SkPipeline(
            [
                ('PolynomialFeatures', PolynomialFeatures(degree)),
                ('Ridge', Ridge(alpha=alpha)),
            ]
        )
        model.fit(X, y)
        ridge_model = model.named_steps['Ridge']

        return {
            Spec.MODEL.name: model,
            Spec.DEGREE.name: degree,
            Spec.ALPHA.name: alpha,
            Spec.COEFFICIENTS.name: ridge_model.coef_,
            Spec.INTERCEPT.name: float(ridge_model.intercept_),
        }


@step(
    id='PL-PRR',
    description='Generates plots for Polynomial Ridge Regression model '
    'results.',
    input_specs=[
        Spec.ON.optional(Spec.DATASET.name).desc('Dataset to plot against.'),
        Spec.MODEL.desc('Trained polynomial ridge regression model'),
        Spec.X_MEAN,
        Spec.X_STD,
        Spec.Y_MEAN.optional(0.0),
        Spec.Y_STD.optional(1.0),
        Spec.DATASET_SCHEMA,
        Spec.PRED_DATASET.optional(),
        Spec.CURVE_EXTEND_MARGIN_RATIO.optional(0.05),
        Spec.TEST_SET.optional(),
        Spec.DATASET_IS_NORMALIZED.optional(False),
    ],
    output_specs=[Spec.FIGURE, Spec.AXES],
)
class PolynomialRidgeRegressionPlotter(
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
        dataset_is_normalized: bool = False,
    ) -> IOValueMap:
        from sklearn.pipeline import Pipeline as SkPipeline

        SAMPLE_POINT_SIZE = 60
        ORIGINAL_SAMPLE_COLOR = 'cornflowerblue'
        OUTLIER_SAMPLE_COLOR = 'red'
        PREDICTED_SAMPLE_COLOR = 'green'
        LINE_COLOR = 'orange'
        LINE_LABEL = 'Polynomial Ridge Fit'

        self.verify_model(model, SkPipeline)

        X_true, y_true = self.get_dataset_from_on(on)
        self.check_feature_dimension(X_true, expected_dimension=1)

        x_mean_scalar = float(x_mean[0])
        x_std_scalar = float(x_std[0])
        y_mean_scalar = float(y_mean)
        y_std_scalar = float(y_std)

        def denormalize_dataset(dataset: Dataset) -> Dataset:
            x_data, y_data = dataset
            return (
                x_data * x_std_scalar + x_mean_scalar,
                y_data * y_std_scalar + y_mean_scalar,
            )

        if dataset_is_normalized:
            X_true, y_true = denormalize_dataset((X_true, y_true))

        x_true = X_true.ravel()

        # ---- Create x_all for later use
        x_all = x_true
        if pred_dataset is not None:
            if dataset_is_normalized:
                x_pred, y_pred = pred_dataset
                pred_dataset = (
                    x_pred * x_std_scalar + x_mean_scalar,
                    y_pred * y_std_scalar + y_mean_scalar,
                )
            x_pred, y_pred = pred_dataset
            x_all = np.concatenate([x_true, x_pred.ravel()])
        if test_set is not None:
            if dataset_is_normalized:
                x_test, y_test = test_set
                test_set = (
                    x_test * x_std_scalar + x_mean_scalar,
                    y_test * y_std_scalar + y_mean_scalar,
                )
            x_test, _ = test_set
            x_all = np.concatenate([x_all, x_test.ravel()])

        # ---- Create a plot
        figure, axes = self.create_plot_or_default()

        # ---- Plot original data points (training set)
        self.single_dispatch(
            PlotScatterPlot,
            {
                **self._inputs,
                Spec.ON.name: (X_true, y_true),
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

        # ---- Plot the polynomial ridge regression curve
        x_min, x_max = x_all.min(), x_all.max()
        x_min, x_max = self.extend_range(
            x_min, x_max, curve_extend_margin_ratio
        )

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

        # ---- Overlay outlier points on top of existing plots
        outlier_dataset = None
        for input_name, value in self._inputs.items():
            if 'outlier_dataset' not in input_name.lower():
                continue
            if (
                isinstance(value, tuple)
                and len(value) == 2
                and isinstance(value[0], np.ndarray)
                and isinstance(value[1], np.ndarray)
            ):
                outlier_dataset = value
                break

        if outlier_dataset is not None and dataset_is_normalized:
            outlier_dataset = denormalize_dataset(outlier_dataset)

        if outlier_dataset is None:
            outlier_indexes_raw = self._inputs.get(Spec.OUTLIER_INDEXES.name)
            if isinstance(outlier_indexes_raw, (list, np.ndarray)):
                outlier_indexes = np.asarray(outlier_indexes_raw, dtype=int)
                outlier_indexes = outlier_indexes[
                    (outlier_indexes >= 0) & (outlier_indexes < len(X_true))
                ]
                if len(outlier_indexes) > 0:
                    outlier_dataset = (
                        X_true[outlier_indexes],
                        y_true[outlier_indexes],
                    )

        if outlier_dataset is not None:
            self.single_dispatch(
                PlotScatterPlot,
                {
                    **self._inputs,
                    Spec.ON.name: outlier_dataset,
                    Spec.FIGURE.name: figure,
                    Spec.AXES.name: axes,
                    Spec.Z_ORDER.name: 3,
                    Spec.SAMPLE_POINT_SIZE.name: SAMPLE_POINT_SIZE,
                    Spec.SAMPLE_POINT_COLOR.name: OUTLIER_SAMPLE_COLOR,
                    Spec.SAMPLE_POINT_LABEL.name: 'Outliers',
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

        X_true, _ = self.get_dataset_from_on(on)
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
