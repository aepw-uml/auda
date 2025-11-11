from auda.dat.models import ModelISName, ModelOSName
from auda.dat.stats import StatOSName
from auda.dat.transformers import TransformerOSName

EVALUATOR_KIND = 'evaluator'


class EvaluatorISName:
    # Samples
    SAMPLES = ModelOSName.SAMPLES
    TRAIN_SAMPLES = TransformerOSName.TRAIN_SAMPLES
    VALIDATION_SAMPLES = TransformerOSName.VALIDATION_SAMPLES
    TEST_SAMPLES = TransformerOSName.TEST_SAMPLES

    # Statistics
    X_MEAN = StatOSName.X_MEAN
    X_STANDARD_DEVIATION = StatOSName.X_STANDARD_DEVIATION

    # Polynomial regression specific
    DEGREE = ModelISName.DEGREE
    INTERCEPT = ModelOSName.INTERCEPT
    COEFFICIENTS = ModelOSName.COEFFICIENTS
    COEFFICIENTS_EXPONENTS = ModelOSName.COEFFICIENTS_EXPONENTS


class EvaluatorOSName:
    # Polynomial regression specific
    Y_TRUE = 'y_true'
    Y_PRED = 'y_pred'
    RESIDUALS = 'residuals'
    R2 = 'r2'
    ROOT_MEAN_SQUARED_ERROR = 'root_mean_squared_error'
    MEAN_ABSOLUTE_ERROR = 'mean_absolute_error'
    MEAN_ABSOLUTE_PERCENTAGE_ERROR = 'mean_absolute_percentage_error'
