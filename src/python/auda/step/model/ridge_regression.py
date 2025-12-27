from typing import override

from auda.step.dataset import DatasetBasedStep
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
