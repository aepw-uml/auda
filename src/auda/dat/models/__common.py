from typing import List, Tuple, cast

import numpy as np

from auda.dat.datasets import DatasetOSName, LabeledSamples
from auda.dat.stats import StatOSName
from auda.dat.transformers import TransformerOSName


class ModelISName:
    # Data
    SAMPLES = DatasetOSName.SAMPLES

    # Standardization specific
    X_STANDARDIZED = TransformerOSName.X_STANDARDIZED
    Y_STANDARDIZED = TransformerOSName.Y_STANDARDIZED

    # Standardization specific (from stats)
    X_MEAN = StatOSName.X_MEAN
    X_STANDARD_DEVIATION = StatOSName.X_STANDARD_DEVIATION
    Y_MEAN = StatOSName.Y_MEAN
    Y_STANDARD_DEVIATION = StatOSName.Y_STANDARD_DEVIATION

    # Normalization specific
    NORMALIZED_SAMPLES = 'normalized_samples'

    # Polynomial regression specific
    DEGREE = 'degree'

    # Support vector regression specific
    REGULARIZATION_PARAMETER = 'regularization_parameter'
    EPSILON = 'epsilon'

    # Anomaly detection specific
    INLIER_SAMPLES = 'inlier_samples'


class ModelOSName:
    SAMPLES = DatasetOSName.SAMPLES

    # Model
    MODEL_TYPE = 'model_type'
    MODEL = 'model'

    # Polynomial regression and other models
    INTERCEPT = 'intercept'
    COEFFICIENTS = 'coefficients'

    # Support vector regression specific
    REGULARIZATION_PARAMETER = ModelISName.REGULARIZATION_PARAMETER
    EPSILON = ModelISName.EPSILON
    SUPPORT_VECTORS_COUNT = 'support_vectors_count'
    SUPPORT_INDICES = 'support_indices'
    DUAL_COEFFICIENTS = 'dual_coefficients'
    GAMMA = 'gamma'

    # Correlation specific
    CORRELATION_MATRIX = 'correlation_matrix'

    # Anomaly detection specific
    INLIER_SAMPLES = ModelISName.INLIER_SAMPLES
    OUTLIER_INDICES = 'outlier_indices'
    INLIER_INDICES = 'inlier_indices'
    CONTAINATION_RATE = 'contamination_rate'

    # Isolation forest specific
    ANOMALY_SCORES = 'norm_scores'

    # Random forest specific
    FEATURE_IMPORTANCES = 'feature_importances'


MODEL_KIND = 'model'


class ModelType:
    POLYNOMIAL_REGRESSION = 'PolynomialRegression'
    SUPPORT_VECTOR_REGRESSION = 'SupportVectorRegression'
    ISOLATION_FOREST = 'IsolationForest'
    RANDOM_FOREST = 'RandomForest'


# The curve is represented as a list of labeled samples
Curve = LabeledSamples


def split_labeled_samples(samples: LabeledSamples) -> Tuple[np.ndarray, np.ndarray]:
    x = np.array([x for x, _ in samples])
    y = np.array([y for _, y in samples])

    return x, y


def create_curve(x_curve: List[float | List[float]], y_curve: List[float]) -> Curve:
    if len(x_curve) != len(y_curve):
        raise ValueError('x_curve and y_curve must have the same length.')

    def to_float_list(x: float | List[float]) -> List[float]:
        return [x] if isinstance(x, float) else cast(List[float], x)

    return [(to_float_list(x), y) for x, y in zip(x_curve, y_curve)]
