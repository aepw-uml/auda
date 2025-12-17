from typing import List, Tuple, Union

from auda.utils.pipeline import IOSpec
from numpy import ndarray

Dataset = Union[Tuple[ndarray, ndarray], ndarray]


class Spec(str, IOSpec):
    # =========================================================================#
    # Samples/Dataset                                                          #
    # =========================================================================#

    DATASET = IOSpec(
        name='DATASET',
        description='Dataset containing the samples.',
        dtype=Dataset,
    )

    NUM_SAMPLES = IOSpec(
        name='NUM_SAMPLES',
        description='Number of samples',
        dtype=int,
    )

    NUM_FEATURES = IOSpec(
        name='NUM_FEATURES',
        description='Number of features',
        dtype=int,
    )

    FEATURE_NAMES = IOSpec(
        name='FEATURE_NAMES',
        description='Names of the features',
        dtype=List[str],
    )

    LABEL_NAMES = IOSpec(
        name='LABEL_NAMES',
        description='Names of the labels',
        dtype=List[str],
    )

    TARGET_CLASSES = IOSpec(
        name='TARGET_CLASSES',
        description='Names of the target classes',
        dtype=List[str],
    )

    FEATURE_UNITS = IOSpec(
        name='FEATURE_UNITS',
        description='Units of the features',
        dtype=List[str],
    )

    LABEL_UNITS = IOSpec(
        name='LABEL_UNITS',
        description='Units of the labels',
        dtype=List[str],
    )

    # =========================================================================#
    # Statistics                                                               #
    # =========================================================================#

    X_MEAN = IOSpec(
        name='X_MEAN',
        description='Mean of the features',
        dtype=ndarray,
    )

    X_STD = IOSpec(
        name='X_STD',
        description='Standard deviation of the features',
        dtype=ndarray,
    )

    X_MINIMUM = IOSpec(
        name='X_MINIMUM',
        description='Minimum values of the features',
        dtype=ndarray,
    )

    X_MAXIMUM = IOSpec(
        name='X_MAXIMUM',
        description='Maximum values of the features',
        dtype=ndarray,
    )

    Y_MEAN = IOSpec(
        name='Y_MEAN',
        description='Mean of the labels',
        dtype=ndarray,
    )

    Y_STD = IOSpec(
        name='Y_STD',
        description='Standard deviation of the labels',
        dtype=ndarray,
    )

    Y_MINIMUM = IOSpec(
        name='Y_MINIMUM',
        description='Minimum values of the labels',
        dtype=ndarray,
    )

    Y_MAXIMUM = IOSpec(
        name='Y_MAXIMUM',
        description='Maximum values of the labels',
        dtype=ndarray,
    )

    # =========================================================================#
    # Splitting Data                                                           #
    # =========================================================================#

    SPLIT_SHUFFLE = IOSpec(
        name='SPLIT_SHUFFLE',
        description='Whether to shuffle the samples before splitting',
        dtype=bool,
    )

    SPLIT_SHUFFLE_SEED = IOSpec(
        name='SPLIT_SHUFFLE_SEED',
        description='The random seed for shuffling the samples before '
        'splitting',
        dtype=int,
    )

    TRAINIING_SET = IOSpec(
        name='TRAINING_SET',
        description='The training set',
        dtype=Dataset,
    )

    VALIDATION_SET = IOSpec(
        name='VALIDATION_SET',
        description='The validation set',
        dtype=Dataset,
    )

    TEST_SET = IOSpec(
        name='TEST_SET',
        description='The test set',
        dtype=Dataset,
    )

    TRAINING_SET_PROPORTION = IOSpec(
        name='TRAINING_SET_PROPORTION',
        description='The proportion of the training set',
        dtype=float,
    )

    VALIDATION_SET_PROPORTION = IOSpec(
        name='VALIDATION_SET_PROPORTION',
        description='The proportion of the validation set',
        dtype=float,
    )

    NUM_TRAINING_SAMPLES = IOSpec(
        name='NUM_TRAINING_SAMPLES',
        description='The number of samples in the training set',
        dtype=int,
    )

    NUM_VALIDATION_SAMPLES = IOSpec(
        name='NUM_VALIDATION_SAMPLES',
        description='The number of samples in the validation set',
        dtype=int,
    )

    NUM_TEST_SAMPLES = IOSpec(
        name='NUM_TEST_SAMPLES',
        description='The number of samples in the test set',
        dtype=int,
    )

    # =========================================================================#
    # Transformation                                                           #
    # =========================================================================#

    NORMALIZED_DATASET = IOSpec(
        name='NORMALIZED_DATASET',
        description='Normalized dataset',
        dtype=Dataset,
    )

    STANDARDIZE_Y = IOSpec(
        name='STANDARDIZE_Y',
        description='Whether to standardize the labels',
        dtype=bool,
    )

    NORM_SCALER_MIN = IOSpec(
        name='NORM_SCALER_MIN',
        description='Normalization scaler minimum value',
        dtype=ndarray,
    )

    NORM_SCALER_MAX = IOSpec(
        name='NORM_SCALER_MAX',
        description='Normalization scaler maximum value',
        dtype=ndarray,
    )

    # =========================================================================#
    # Dataset Specific                                                         #
    # =========================================================================#

    LOCATION = IOSpec(
        name='LOCATION',
        description='The location associated with the samples',
        dtype=List[str],
    )

    YEAR = IOSpec(
        name='YEAR',
        description='The year associated with the samples',
        dtype=List[int],
    )

    # =========================================================================#
    # Reference                                                                #
    # =========================================================================#

    ON = IOSpec(
        name='ON',
        description='Tell the step which split to use',
        dtype=str,
    )
