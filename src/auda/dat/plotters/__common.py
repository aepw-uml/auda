from typing import Tuple

from auda.dat.datasets import DatasetOSName
from auda.dat.models import ModelISName, ModelOSName
from auda.dat.predictors import PredictorOSName
from auda.dat.stats import StatOSName


class PlotterISName:
    # General
    TITLE = 'title'
    SAMPLES = DatasetOSName.SAMPLES
    LABEL = DatasetOSName.LABEL
    FEATURE_NAMES = DatasetOSName.FEATURE_NAMES
    UNITS = DatasetOSName.UNITS

    # Figure specific
    FIGURE = 'figure'
    SAVE_PATH = 'save_path'
    SAVE_DPI = 'save_dpi'
    SAVE_TRANSPARENT = 'save_transparent'

    # Stats specific
    X_MEAN = StatOSName.X_MEAN
    X_STANDARD_DEVIATION = StatOSName.X_STANDARD_DEVIATION
    Y_MEAN = StatOSName.Y_MEAN
    Y_STANDARD_DEVIATION = StatOSName.Y_STANDARD_DEVIATION

    # Model
    MODEL_TYPE = ModelOSName.MODEL_TYPE
    MODEL = ModelOSName.MODEL

    # Prediction specific
    PREDICTION_SAMPLES = PredictorOSName.PREDICTION_SAMPLES

    # Polynomial Regression specific
    INTERCEPT = ModelOSName.INTERCEPT
    COEFFICIENTS = ModelOSName.COEFFICIENTS

    # Support Vector Regression specific
    EPSILON = ModelISName.EPSILON
    SUPPORT_INDICES = ModelOSName.SUPPORT_INDICES

    # Correlation specific
    CORRELATION_MATRIX = ModelOSName.CORRELATION_MATRIX

    # Anomaly Detection specific
    INLIER_SAMPLES = ModelOSName.INLIER_SAMPLES

    # Feature Importances specific
    FEATURE_IMPORTANCES = ModelOSName.FEATURE_IMPORTANCES


class PlotterOSName:
    # Figure
    FIGURE = PlotterISName.FIGURE


PLOTTER_KIND = 'plotter'


def extend_range(
    lower_bound: float,
    upper_bound: float,
    margin_ratio: float = 0.05,
) -> Tuple[float, float]:
    """
    Expands a numeric range by a relative margin on both sides.

    Args:
        lower_bound: The minimum x-value of the range.
        upper_bound: The maximum x-value of the range.
        margin_ratio: Fraction of the range to extend on each side (default: 0.05 → 5%).

    Returns:
        A tuple of (extended_lower, extended_upper).
    """
    span = upper_bound - lower_bound
    margin = margin_ratio * span
    return lower_bound - margin, upper_bound + margin
