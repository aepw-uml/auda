from typing import override

from auda.step.dataset import DatasetBasedStep
from auda.step.spec import Dataset, Spec
from auda.utils.pipeline import IOValueMap, step


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
