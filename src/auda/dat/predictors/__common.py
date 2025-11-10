from auda.dat.datasets import DatasetOSName
from auda.dat.models import ModelISName, ModelOSName
from auda.dat.stats import StatOSName
from auda.utils.pipeline import Task


class PredictorISName:
    SAMPLES = DatasetOSName.SAMPLES
    X_PREDICT = 'x_predict'

    # Stats specific
    X_MEAN = StatOSName.X_MEAN
    X_STANDARD_DEVIATION = StatOSName.X_STANDARD_DEVIATION
    Y_MEAN = StatOSName.Y_MEAN
    Y_STANDARD_DEVIATION = StatOSName.Y_STANDARD_DEVIATION

    # Model specific
    MODEL_TYPE = ModelOSName.MODEL_TYPE
    MODEL = ModelOSName.MODEL

    # Support Vector Regression hyperparameters
    X_MINIMUM = StatOSName.X_MINIMUM
    X_MAXIMUM = StatOSName.X_MAXIMUM
    REGULARIZATION_PARAMETER = ModelISName.REGULARIZATION_PARAMETER
    GAMMA = ModelOSName.GAMMA


class PredictorOSName:
    PREDICTION_SAMPLES = 'prediction_samples'


PREDICTOR_KIND = 'predictor'


def verify_previous_model(task: Task, expected_model_type: str) -> None:
    model_type = task.get_input(PredictorISName.MODEL_TYPE)

    if model_type is None:
        raise ValueError('No model type found in the task inputs.')

    if model_type != expected_model_type:
        raise ValueError(
            f'Invalid model type: expected {expected_model_type}, got {model_type}.'
        )
