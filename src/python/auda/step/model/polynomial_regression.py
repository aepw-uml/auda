from typing import override

from auda.step.dataset import DatasetBasedStep
from auda.step.model import ModelBasedStep
from auda.step.plot import PlotStep
from auda.step.spec import Dataset, Spec
from auda.utils.pipeline import IOValueMap, step
from sklearn.base import np


@step(
    id='MD-PR',
    description='Polynomial Regression Model Step',
    input_specs=[Spec.ON.optional(Spec.DATASET.name), Spec.DEGREE.optional(3)],
    output_specs=[Spec.MODEL, Spec.DEGREE, Spec.INTERCEPT, Spec.COEFFICIENTS],
)
class PolynomialRegressionModel(DatasetBasedStep):
    @override
    def run(self, on: str | Dataset, degree: int) -> IOValueMap:
        from sklearn.linear_model import LinearRegression
        from sklearn.pipeline import Pipeline as SkPipeline
        from sklearn.preprocessing import PolynomialFeatures

        X, y = self.get_dataset_from_on(on)

        model = SkPipeline(
            [
                ('PolynomialFeatures', PolynomialFeatures(degree)),
                ('LinearRegression', LinearRegression()),
            ]
        )
        model.fit(X, y)
        linear_regression_model = model.named_steps['LinearRegression']

        return {
            Spec.MODEL.name: model,
            Spec.DEGREE.name: degree,
            Spec.COEFFICIENTS.name: linear_regression_model.coef_,
            Spec.INTERCEPT.name: linear_regression_model.intercept_,
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
class PolynomialRegressionPlotter(PlotStep, DatasetBasedStep, ModelBasedStep):
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
        return {}
