from typing import List, Literal, Tuple, Union

from auda.utils.pipeline import IOSpec, Pipeline
from numpy import ndarray

LabeledDataset = Tuple[ndarray, ndarray]
UnlabeledDataset = ndarray
Dataset = Union[LabeledDataset, UnlabeledDataset]

Interval = Tuple[float, float]


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

    DATASET_SCEHMA = IOSpec(
        name='DATASET_SCEHMA',
        description='Schema of the dataset',
        dtype=dict,
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
        dtype=Union[ndarray, float],
    )

    Y_STD = IOSpec(
        name='Y_STD',
        description='Standard deviation of the labels',
        dtype=Union[ndarray, float],
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

    NUM_K_FOLDS = IOSpec(
        name='NUM_K_FOLDS',
        description='Number of folds for K-Fold Cross Validation',
        dtype=int,
    )

    # =========================================================================#
    # REPRODUCIBILITY                                                          #
    # =========================================================================#

    SEED = IOSpec(
        name='SEED',
        description='The random seed for reproducible results',
        dtype=int,
    )

    # =========================================================================#
    # ANOMALY DETECTION                                                        #
    # =========================================================================#

    ISOLATION_FOREST_THRESHOLD = IOSpec(
        name='ISOLATION_FOREST_THRESHOLD',
        description='Threshold for Isolation Forest to classify anomalies',
        dtype=float,
    )

    INLIER_INDEXES = IOSpec(
        name='INLIER_INDEXES',
        description='Indexes of inlier samples',
        dtype=List[int],
    )

    OUTLIER_INDEXES = IOSpec(
        name='OUTLIER_INDEXES',
        description='Indexes of outlier samples',
        dtype=List[int],
    )

    INLIER_DATASET = IOSpec(
        name='INLIER_DATASET',
        description='Dataset containing inlier samples',
        dtype=Dataset,
    )

    ANOMALY_SCORES = IOSpec(
        name='ANOMALY_SCORES',
        description='Anomaly scores for each sample',
        dtype=ndarray,
    )

    CONTAMINATION_RATE = IOSpec(
        name='CONTAMINATION_RATE',
        description='Proportion of outliers in the dataset',
        dtype=float,
    )

    # =========================================================================#
    # Reference                                                                #
    # =========================================================================#

    ON = IOSpec(
        name='ON',
        description='Tell the step which dataset to use',
        dtype=Union[str, Dataset],
    )

    PIPE = IOSpec(
        name='PIPE',
        description='The sub pipeline to be executed',
        dtype=str | Pipeline,
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
    # Tuning Specific                                                          #
    # =========================================================================#

    METRIC = IOSpec(
        name='METRIC',
        description='Metric used for hyperparameter tuning',
        dtype=Literal['mae', 'rmse', 'r2', 'mape'],
    )

    EXPECT_HIGHER = IOSpec(
        name='EXPECT_HIGHER',
        description='Whether higher metric values are better',
        dtype=bool,
    )

    SEARCH_SPACE = IOSpec(
        name='SEARCH_SPACE',
        description='Sampling intervals for hyperparameter tuning.',
        dtype=List[List[Interval]],
    )

    NUM_ITERATIONS = IOSpec(
        name='NUM_ITERATIONS',
        description='Number of iterations for hyperparameter tuning',
        dtype=int,
    )

    HYPERPARAMETER_NAMES = IOSpec(
        name='HYPERPARAMETER_NAMES',
        description='Names of the hyperparameters being tuned',
        dtype=List[str],
    )

    HYPERPARAMETERS_SCORE_LIST = IOSpec(
        name='HYPERPARAMETERS_SCORE_LIST',
        description='List of hyperparameter sets and their corresponding '
        'scores',
        dtype=List[Tuple[List[float], float]],
    )

    BEST_HYPERPARAMETERS = IOSpec(
        name='BEST_HYPERPARAMETERS',
        description='Best hyperparameters found during tuning',
        dtype=List[float],
    )

    BEST_SCORE = IOSpec(
        name='BEST_SCORE',
        description='Best score achieved during hyperparameter tuning',
        dtype=float,
    )

    HYPERPARAMETER_DOMAINS = IOSpec(
        name='HYPERPARAMETER_DOMAINS',
        description='Domains of the hyperparameters being tuned',
        dtype=List[Interval],
    )

    ELITE_FRACTIONS = IOSpec(
        name='ELITE_FRACTIONS',
        description='Fractions of top-performing hyperparameter sets to '
        'consider as elite for each stage',
        dtype=List[float],
    )

    REFINEMENT_WIDTHS = IOSpec(
        name='REFINEMENT_WIDTHS',
        description='Widths for refining the search space around elite '
        'hyperparameter sets for each stage',
        dtype=List[List[float]],
    )

    HYPERPARAMETERS_SCORE_LISTS = IOSpec(
        name='HYPERPARAMETERS_SCORE_LISTS',
        description='List of hyperparameter sets and their corresponding '
        'scores for each stage',
        dtype=List[List[Tuple[List[float], float]]],
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

    AXES = IOSpec(
        name='AXES',
        description='Matplotlib Axes object',
        dtype=object,
    )

    TITLE = IOSpec(
        name='TITLE',
        description='Title of the figure',
        dtype=str,
    )

    SAMPLE_POINT_SIZE = IOSpec(
        name='SAMPLE_POINT_SIZE',
        description='Size of the sample points in the plot',
        dtype=float,
    )

    SAMPLE_POINT_COLOR = IOSpec(
        name='SAMPLE_POINT_COLOR',
        description='Color of the sample points in the plot',
        dtype=str,
    )

    SAMPLE_POINT_EDGE_COLOR = IOSpec(
        name='SAMPLE_POINT_EDGE_COLOR',
        description='Edge color of the sample points in the plot',
        dtype=str,
    )

    SAMPLE_POINT_LABEL = IOSpec(
        name='SAMPLE_POINT_LABEL',
        description='Label for the sample points in the plot',
        dtype=str,
    )

    LINE_COLOR = IOSpec(
        name='LINE_COLOR',
        description='Color of the line in the plot',
        dtype=str,
    )

    LINE_WIDTH = IOSpec(
        name='LINE_WIDTH',
        description='Width of the line in the plot',
        dtype=float,
    )

    LINE_LABEL = IOSpec(
        name='LINE_LABEL',
        description='Label for the line in the plot',
        dtype=str,
    )

    LINE_STYLE = IOSpec(
        name='LINE_STYLE',
        description='Style of the line in the plot',
        dtype=str,
    )
