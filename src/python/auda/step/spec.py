from typing import List, Tuple, Union

from auda.utils.pipeline import IOSpec
from numpy import ndarray

LabeledDataset = Tuple[ndarray, ndarray]
UnlabeledDataset = ndarray
Dataset = Union[LabeledDataset, UnlabeledDataset]


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
    # Evaluation                                                               #
    # =========================================================================#

    MAE = IOSpec(
        name='MAE',
        description='Mean Absolute Error',
        dtype=float,
    )

    RMSE = IOSpec(
        name='RMSE',
        description='Root Mean Square Error',
        dtype=float,
    )

    R2 = IOSpec(
        name='R2',
        description='R-squared',
        dtype=float,
    )

    MAPE = IOSpec(
        name='MAPE',
        description='Mean Absolute Percentage Error',
        dtype=float,
    )

    # =========================================================================#
    # Reference                                                                #
    # =========================================================================#

    ON = IOSpec(
        name='ON',
        description='Tell the step which split to use',
        dtype=str,
    )

    PIPE = IOSpec(
        name='PIPE',
        description='The processing pipeline associated with the dataset',
        dtype=str,
    )

    # =========================================================================#
    # Model Specific                                                           #
    # =========================================================================#

    MODEL = IOSpec(
        name='MODEL',
        description='The model used in the pipeline.',
        dtype=object,
    )

    TRAINING_MODEL = IOSpec(
        name='TRAINING_MODEL',
        description='The trained model after fitting to the training data.',
        dtype=object,
    )

    KERNEL = IOSpec(
        name='KERNEL',
        description='Kernel type',
        dtype=str,
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
    # Regression Models Specific                                               #
    # =========================================================================#

    C = IOSpec(
        name='C',
        description='Regularization parameter for regression models',
        dtype=float,
    )

    EPSILON = IOSpec(
        name='EPSILON',
        description='Epsilon parameter for Support Vector Regression (SVR)',
        dtype=float,
    )

    NUM_SUPPORT_VECTORS = IOSpec(
        name='NUM_SUPPORT_VECTORS',
        description='Number of support vectors in Support Vector Regression '
        '(SVR) model',
        dtype=int,
    )

    DUAL_COEFFICIENTS = IOSpec(
        name='DUAL_COEFFICIENTS',
        description='Dual coefficients of the support vectors in Support '
        'Vector Regression (SVR) model',
        dtype=List[float],
    )

    INTERCEPT = IOSpec(
        name='INTERCEPT',
        description='Intercept of the Support Vector Regression (SVR) model',
        dtype=float,
    )

    GAMMA = IOSpec(
        name='GAMMA',
        description='Gamma parameter of the Support Vector Regression (SVR) '
        'model',
        dtype=float,
    )

    # =========================================================================#
    # Matplotlib Specific                                                      #
    # =========================================================================#

    FIGURE = IOSpec(
        name='FIGURE',
        description='Generated figure',
        dtype=object,
    )

    TITLE = IOSpec(
        name='TITLE',
        description='Title of the figure',
        dtype=str,
    )
