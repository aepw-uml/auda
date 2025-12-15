import numpy as np

from auda.dat.datasets import DatasetOSName
from auda.dat.stats import StatOSName
from auda.utils.pipeline import Task


class TransformerISName:
    SAMPLES = DatasetOSName.SAMPLES
    ORIGINAL_SAMPLES = DatasetOSName.ORIGINAL_SAMPLES

    # Stardardization specific
    TO_STANDARDIZE_Y = 'to_standardize_y'

    # Min Max Scaler specific
    SCALER_MIN = 'scaler_min'
    SCALER_MAX = 'scaler_max'

    # Trimming specific
    X_MINIMUM = StatOSName.X_MINIMUM
    X_MAXIMUM = StatOSName.X_MAXIMUM
    LOWER_TRIMMING_PERCENTAGE = 'lower_trimming_percentage'
    UPPER_TRIMMING_PERCENTAGE = 'upper_trimming_percentage'

    # Split specific
    SPLIT_SHUFFLE = 'split_shuffle'
    SPLIT_SHUFFLE_SEED = 'split_seed'
    TRAINING_FRACTION = 'training_fraction'
    VALIDATION_FRACTION = 'validation_fraction'

    TEST_SAMPLES = 'test_samples'
    X_MEAN = 'x_mean'
    X_STANDARD_DEVIATION = 'x_standard_deviation'


class TransformerOSName:
    SAMPLES = DatasetOSName.SAMPLES

    # Standardization specific
    X_STANDARDIZED = 'x_standardized'
    Y_STANDARDIZED = 'y_standardized'

    # Scaler specific
    SCALED_SAMPLES = 'scaled_samples'

    # Normalization specific
    NORMALIZED_SAMPLES = 'normalized_samples'

    # Trimming specific
    TRIMMED_SAMPLES = 'trimmed_samples'

    # Split specific
    TRAINING_SAMPLES = 'train_samples'
    VALIDATION_SAMPLES = 'validation_samples'
    TEST_SAMPLES = 'test_samples'
    NUM_TRAIN_SAMPLES = 'num_train_samples'
    NUM_VALIDATION_SAMPLES = 'num_validation_samples'
    NUM_TEST_SAMPLES = 'num_test_samples'

    ORIGINAL_TEST_SAMPLES = ''
    TEST_SAMPLES = ''


TRANSFORMER_KIND = 'transformer'


def standardize_x(task: Task, x: np.ndarray) -> np.ndarray:
    """
    Standardize the feature values using stored statistics in the task.

    Args:
        task: Task instance providing mean and standard deviation via StatOSName.
        x: Array of raw feature values.

    Returns:
        np.ndarray: Standardized values (zero mean, unit variance).
    """
    x_mean = np.asarray(task.get_input(StatOSName.X_MEAN), dtype=float).ravel()
    x_std_dev = np.asarray(
        task.get_input(StatOSName.X_STANDARD_DEVIATION), dtype=float
    ).ravel()

    return (x - x_mean) / x_std_dev


def reverse_x(task: Task, x_std: np.ndarray) -> np.ndarray:
    """
    Reverse the standardization of feature values back to original scale.

    Args:
        task: Task instance providing mean and standard deviation via StatOSName.
        x_std: Standardized feature values.

    Returns:
        np.ndarray: Reconstructed raw feature values.
    """
    x_mean = np.asarray(task.get_input(StatOSName.X_MEAN), dtype=float).ravel()
    x_std_dev = np.asarray(
        task.get_input(StatOSName.X_STANDARD_DEVIATION), dtype=float
    ).ravel()

    return x_std * x_std_dev + x_mean


def standardize_y(task: Task, y: np.ndarray) -> np.ndarray:
    """
    Standardize target variable values using stored statistics in the task.

    Args:
        task: Task instance providing mean and standard deviation via StatOSName.
        y: Array of raw target values.

    Returns:
        np.ndarray: Standardized target values (zero mean, unit variance).
    """
    y_mean = float(task.get_input(StatOSName.Y_MEAN))
    y_std_dev = float(task.get_input(StatOSName.Y_STANDARD_DEVIATION))

    return (y - y_mean) / y_std_dev


def reverse_y(task: Task, y_std: np.ndarray) -> np.ndarray:
    """
    Reverse the standardization of target variable values back to original scale.

    Args:
        task: Task instance providing mean and standard deviation via StatOSName.
        y_std: Standardized target values.

    Returns:
        np.ndarray: Reconstructed raw target values.
    """
    y_mean = float(task.get_input(StatOSName.Y_MEAN))
    y_std_dev = float(task.get_input(StatOSName.Y_STANDARD_DEVIATION))

    return y_std * y_std_dev + y_mean
