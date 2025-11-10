from typing import override

from auda.dat.datasets import LabeledSamples
from auda.dat.models import ModelType
from auda.utils.pipeline import IOSpec, Task, task

from .__common import (
    PREDICTOR_KIND,
    PredictorISName,
    PredictorOSName,
    verify_previous_model,
)


@task(
    id='PD-SVR',
    kind=PREDICTOR_KIND,
    description='Generates predictions using a trained Support Vector Regression'
    '(SVR) model.',
    input_specs={
        PredictorISName.MODEL_TYPE: IOSpec(dtype=str),
        PredictorISName.MODEL: IOSpec(dtype=object),
        PredictorISName.X_PREDICT: IOSpec(dtype=str),
        PredictorISName.X_MEAN: IOSpec(dtype=float),
        PredictorISName.X_STANDARD_DEVIATION: IOSpec(dtype=float),
        PredictorISName.Y_MEAN: IOSpec(dtype=float),
        PredictorISName.Y_STANDARD_DEVIATION: IOSpec(dtype=float),
    },
    output_specs={
        PredictorOSName.PREDICTION_SAMPLES: IOSpec(dtype=LabeledSamples),
    },
)
class SupportVectorRegressionPredictor(Task):
    """
    This task predicts using the trained support vector regression model.
    """

    @override
    def run(self) -> None:
        import numpy as np
        from sklearn.svm import SVR

        verify_previous_model(self, ModelType.SUPPORT_VECTOR_REGRESSION)
        svr_model: SVR = self.get_input(PredictorISName.MODEL)

        x_predict = self.get_input(PredictorISName.X_PREDICT)
        x_pred = np.array([float(elem.strip()) for elem in x_predict.split(',')])

        # ---- Standardize x_pred
        x_mean = self.get_input(PredictorISName.X_MEAN)[0]
        x_std = self.get_input(PredictorISName.X_STANDARD_DEVIATION)[0]
        x_pred_std = (x_pred - x_mean) / x_std

        # ---- Create prediction samples
        y_pred_std = svr_model.predict(x_pred_std.reshape(-1, 1))
        prediction_samples: LabeledSamples = list(
            zip(x_pred_std.tolist(), y_pred_std.tolist())
        )

        # ---- Reverse standardization for predictions
        y_mean = self.get_input(PredictorISName.Y_MEAN)
        y_std = self.get_input(PredictorISName.Y_STANDARD_DEVIATION)
        y_pred = y_pred_std * y_std + y_mean
        prediction_samples = [([x], y) for x, y in zip(x_pred, y_pred)]

        # ---- Populate outputs
        self.set_output(PredictorOSName.PREDICTION_SAMPLES, prediction_samples)
