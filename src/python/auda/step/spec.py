from enum import Enum
from typing import List, Tuple, Union

from auda.utils.pipeline import IOSpec
from numpy import ndarray

Dataset = Union[Tuple[ndarray, ndarray], ndarray]


class SpecName(str, Enum):
    # =========================================================================#
    # Samples/Dataset                                                          #
    # =========================================================================#

    # The dataset
    DATASET = 'DATASET'

    # The number of samples in the dataset
    NUM_SAMPLES = 'NUM_SAMPLES'

    # The number of features in the dataset
    NUM_FEATURES = 'NUM_FEATURES'

    # Names of the features
    FEATURE_NAMES = 'FEATURE_NAMES'

    # Names of the label in regression tasks
    LABEL_NAMES = 'LABEL_NAMES'

    # Names of the target classes in classification tasks
    TARGET_CLASSES = 'TARGET_CLASSES'

    # Units of the features
    FEATURE_UNITS = 'FEATURE_UNITS'

    # Units of the labels
    LABEL_UNITS = 'LABEL_UNITS'

    # =========================================================================#
    # Statistics                                                               #
    # =========================================================================#

    # Mean of the features
    X_MEAN = 'X_MEAN'

    # Standard deviation of the features
    X_STD = 'X_STD'

    # Minimum values of the features
    X_MINIMUM = 'X_MINIMUM'

    # Maximum values of the features
    X_MAXIMUM = 'X_MAXIMUM'

    # Mean of the targets
    Y_MEAN = 'Y_MEAN'

    # Standard deviation of the targets
    Y_STD = 'Y_STD'

    # Minimum values of the targets
    Y_MINIMUM = 'Y_MINIMUM'

    # Maximum values of the targets
    Y_MAXIMUM = 'Y_MAXIMUM'

    # =========================================================================#
    # Splitting Data                                                           #
    # =========================================================================#

    # Whether to shuffle the samples before splitting
    SPLIT_SHUFFLE = 'SPLIT_SHUFFLE'

    # The random seed for shuffling the samples before splitting
    SPLIT_SHUFFLE_SEED = 'SPLIT_SHUFFLE_SEED'

    # The training set
    TRAINIING_SET = 'TRAINING_SET'

    # The validation set
    VALIDATION_SET = 'VALIDATION_SET'

    # The test set
    TEST_SET = 'TEST_SET'

    # The proportion of the training set
    TRAINING_SET_PROPORTION = 'TRAINING_SET_PROPORTION'

    # The proportion of the validation set
    VALIDATION_SET_PROPORTION = 'VALIDATION_SET_PROPORTION'

    # The number of samples in the training set
    NUM_TRAINING_SAMPLES = 'NUM_TRAINING_SAMPLES'

    # The number of samples in the validation set
    NUM_VALIDATION_SAMPLES = 'NUM_VALIDATION_SAMPLES'

    # The number of samples in the test set
    NUM_TEST_SAMPLES = 'NUM_TEST_SAMPLES'

    # =========================================================================#
    # Transformation                                                           #
    # =========================================================================#

    # Whether to standardize the labels (regression tasks)
    STANDARDIZE_LABELS = 'STANDARDIZE_LABELS'

    # Normalization scaler minimum value
    NORM_SCALER_MIN = 'NORM_SCALER_MIN'

    # Normalization scaler maximum value
    NORM_SCALER_MAX = 'NORM_SCALER_MAX'

    # =========================================================================#
    # Dataset Specific                                                         #
    # =========================================================================#

    # The location associated with the samples
    LOCATION = 'LOCATION'

    # The year associated with the samples
    YEAR = 'YEAR'

    # =========================================================================#
    # Reference                                                                #
    # =========================================================================#

    # Tell the step which split to use
    ON = 'ON'


class Spec(str, IOSpec):
    # =========================================================================#
    # Samples/Dataset                                                          #
    # =========================================================================#

    DATASET = IOSpec(
        name=SpecName.DATASET,
        description='Dataset containing the samples.',
        dtype=Dataset,
    )

    NUM_SAMPLES = IOSpec(
        name=SpecName.NUM_SAMPLES,
        description='Number of samples',
        dtype=int,
    )

    NUM_FEATURES = IOSpec(
        name=SpecName.NUM_FEATURES,
        description='Number of features',
        dtype=int,
    )

    FEATURE_NAMES = IOSpec(
        name=SpecName.FEATURE_NAMES,
        description='Names of the features',
        dtype=List[str],
    )

    LABEL_NAMES = IOSpec(
        name=SpecName.LABEL_NAMES,
        description='Names of the labels',
        dtype=List[str],
    )

    TARGET_CLASSES = IOSpec(
        name=SpecName.TARGET_CLASSES,
        description='Names of the target classes',
        dtype=List[str],
    )

    FEATURE_UNITS = IOSpec(
        name=SpecName.FEATURE_UNITS,
        description='Units of the features',
        dtype=List[str],
    )

    LABEL_UNITS = IOSpec(
        name=SpecName.LABEL_UNITS,
        description='Units of the labels',
        dtype=List[str],
    )

    # =========================================================================#
    # Statistics                                                               #
    # =========================================================================#

    X_MEAN = IOSpec(
        name=SpecName.X_MEAN,
        description='Mean of the features',
        dtype=ndarray,
    )

    X_STD = IOSpec(
        name=SpecName.X_STD,
        description='Standard deviation of the features',
        dtype=ndarray,
    )

    X_MINIMUM = IOSpec(
        name=SpecName.X_MINIMUM,
        description='Minimum values of the features',
        dtype=ndarray,
    )

    X_MAXIMUM = IOSpec(
        name=SpecName.X_MAXIMUM,
        description='Maximum values of the features',
        dtype=ndarray,
    )

    Y_MEAN = IOSpec(
        name=SpecName.Y_MEAN,
        description='Mean of the labels',
        dtype=ndarray,
    )

    Y_STD = IOSpec(
        name=SpecName.Y_STD,
        description='Standard deviation of the labels',
        dtype=ndarray,
    )

    Y_MINIMUM = IOSpec(
        name=SpecName.Y_MINIMUM,
        description='Minimum values of the labels',
        dtype=ndarray,
    )

    Y_MAXIMUM = IOSpec(
        name=SpecName.Y_MAXIMUM,
        description='Maximum values of the labels',
        dtype=ndarray,
    )

    # =========================================================================#
    # Splitting Data                                                           #
    # =========================================================================#

    SPLIT_SHUFFLE = IOSpec(
        name=SpecName.SPLIT_SHUFFLE,
        description='Whether to shuffle the samples before splitting',
        dtype=bool,
    )

    SPLIT_SHUFFLE_SEED = IOSpec(
        name=SpecName.SPLIT_SHUFFLE_SEED,
        description='The random seed for shuffling the samples before '
        'splitting',
        dtype=int,
    )

    TRAINIING_SET = IOSpec(
        name=SpecName.TRAINIING_SET,
        description='The training set',
        dtype=Dataset,
    )

    VALIDATION_SET = IOSpec(
        name=SpecName.VALIDATION_SET,
        description='The validation set',
        dtype=Dataset,
    )

    TEST_SET = IOSpec(
        name=SpecName.TEST_SET,
        description='The test set',
        dtype=Dataset,
    )

    TRAINING_SET_PROPORTION = IOSpec(
        name=SpecName.TRAINING_SET_PROPORTION,
        description='The proportion of the training set',
        dtype=float,
    )

    VALIDATION_SET_PROPORTION = IOSpec(
        name=SpecName.VALIDATION_SET_PROPORTION,
        description='The proportion of the validation set',
        dtype=float,
    )

    NUM_TRAINING_SAMPLES = IOSpec(
        name=SpecName.NUM_TRAINING_SAMPLES,
        description='The number of samples in the training set',
        dtype=int,
    )

    NUM_VALIDATION_SAMPLES = IOSpec(
        name=SpecName.NUM_VALIDATION_SAMPLES,
        description='The number of samples in the validation set',
        dtype=int,
    )

    NUM_TEST_SAMPLES = IOSpec(
        name=SpecName.NUM_TEST_SAMPLES,
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
        name=SpecName.NORM_SCALER_MIN,
        description='Normalization scaler minimum value',
        dtype=ndarray,
    )

    NORM_SCALER_MAX = IOSpec(
        name=SpecName.NORM_SCALER_MAX,
        description='Normalization scaler maximum value',
        dtype=ndarray,
    )

    # =========================================================================#
    # Dataset Specific                                                         #
    # =========================================================================#

    LOCATION = IOSpec(
        name=SpecName.LOCATION,
        description='The location associated with the samples',
        dtype=List[str],
    )

    YEAR = IOSpec(
        name=SpecName.YEAR,
        description='The year associated with the samples',
        dtype=List[int],
    )

    # =========================================================================#
    # Reference                                                                #
    # =========================================================================#

    ON = IOSpec(
        name=SpecName.ON,
        description='Tell the step which split to use',
        dtype=str,
    )
