from typing import override

from auda.step import create_pipeline_from_pipe
from auda.step.dataset import DatasetBasedStep
from auda.step.evaluator import mae, mape, mse, r2
from auda.step.spec import Dataset, Spec
from auda.utils.pipeline import (
    IOValueMap,
    Pipeline,
    step,
)


@step(
    id='EV-NORMAL',
    description='Evaluates regression models using standard metrics on a '
    + 'test dataset.',
    input_specs=[
        Spec.ON.optional(Spec.DATASET.name),
        Spec.SEED.optional(42),
        Spec.PIPE,
        Spec.TRAINIING_SET,
        Spec.TEST_SET,
    ],
    output_specs=[Spec.MAE, Spec.RMSE, Spec.R2, Spec.MAPE],
)
class NormalEvaluator(DatasetBasedStep):
    @override
    def run(self, pipe: str | Pipeline, test_set: Dataset) -> IOValueMap:
        # Create a pipeline
        pipeline = create_pipeline_from_pipe(pipe)

        pipeline.run(
            {
                **self._inputs,
                Spec.ON.name: Spec.TRAINIING_SET.name,
            }
        )
        model = pipeline.get_value(Spec.MODEL.name)

        X_test, y_test = test_set
        X_mean = pipeline.get_value(Spec.X_MEAN.name)
        X_std = pipeline.get_value(Spec.X_STD.name)
        X_test_std = (X_test - X_mean[0]) / X_std[0]

        model.predict(X_test_std)
        y_mean = pipeline.get_value(Spec.Y_MEAN.name)
        y_std = pipeline.get_value(Spec.Y_STD.name)
        y_pred_std = model.predict(X_test_std)
        y_pred = y_pred_std * y_std + y_mean

        return {
            Spec.MAE.name: mae(y_test, y_pred),
            Spec.RMSE.name: mse(y_test, y_pred) ** 0.5,
            Spec.R2.name: r2(y_test, y_pred),
            Spec.MAPE.name: mape(y_test, y_pred),
        }
