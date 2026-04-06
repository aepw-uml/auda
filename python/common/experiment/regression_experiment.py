from typing import cast, override

import numpy as np
from common.metrics import RegressionMetrics
from sklearn.metrics import (
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_squared_error,
    r2_score,
)
from step.model.isolation_forest import isolation_forest

from .experiment import Experiment


class RegressionExperiment(Experiment):
    """Base implementation for regression experiments.

    This class stores regression datasets, creates randomized splits, and
    computes standard regression metrics. Concrete subclasses still define the
    actual model training and tuning behavior.

    Attributes:
        X: Full feature matrix provided in ``setup()``.
        y: Full target vector provided in ``setup()``.
        X_train: Training feature matrix created in ``split()``.
        y_train: Training target vector created in ``split()``.
        X_test: Test feature matrix created in ``split()``.
        y_test: Test target vector created in ``split()``.
    """

    @override
    def __init__(
        self,
        name: str,
        description: str,
        train_size: float = 0.9,
        seed: int = 417,
    ) -> None:
        """Initializes a regression experiment.

        The instance starts without any dataset assigned, so ``setup()`` must
        run before the experiment can be split, trained, tuned, or evaluated.
        """

        super().__init__(name, description, train_size, seed)

        self.X: np.ndarray | None = None
        self.y: np.ndarray | None = None
        self.X_train: np.ndarray | None = None
        self.y_train: np.ndarray | None = None
        self.X_test: np.ndarray | None = None
        self.y_test: np.ndarray | None = None

    @override
    def anomaly_detection(self) -> None:
        self.log('Running anomaly detection...')

        if self.X is None or self.y is None:
            raise ValueError('Data not set up. Call setup() first.')

        contamination = float(self.context.get('anomaly_contamination', 0.05))

        result = isolation_forest(self.X, self.y, contamination, self.seed)
        self.context['isolation_forest_result'] = result
        self.context['X_original'] = self.X
        self.context['y_original'] = self.y

        self.X = result.X_inliers
        self.y = result.y_inliers

        # Log the number of samples removed and remaining after anomaly
        # detection.
        num_removed = np.sum(~result.inlier_mask)
        num_remaining = np.sum(result.inlier_mask)
        self.log(
            f'Anomaly detection removed {num_removed} samples, '
            f'leaving {num_remaining} samples for training.'
        )

    @override
    def setup(self, X: np.ndarray, y: np.ndarray, **kwargs) -> None:
        """Stores feature and target arrays for the experiment.

        Subclasses may extend this method to perform additional validation or
        preprocessing, but they should preserve the contract that later
        lifecycle methods can access the raw dataset from instance state.
        Additional keyword arguments are forwarded to ``Experiment.setup()``
        and merged into ``self.context``.

        Args:
            X: Feature matrix.
            y: Target vector.
            **kwargs: Additional experiment metadata or configuration to store
                in ``self.context``.
        """

        self.X = X
        self.y = y
        super().setup(**kwargs)

        m, d = X.shape
        self.log(f'Set up experiment with {m} samples and {d} features.')

    @override
    def split(self) -> None:
        """Randomly splits the dataset into train, validation, and test sets.

        The shuffled splits are written to ``X_train``, ``y_train``, ``X_test``,
        and ``y_test`` for downstream use by ``train()``, ``tune()``, and
        ``evaluate()``.

        Raises:
            ValueError: If the dataset has not been provided via ``setup()``.
        """

        if self.X is None or self.y is None:
            raise ValueError('Data not set up. Call setup() first.')

        enable_evaluation = (
            self.context.get('evaluation.enabled', 'False').lower() == 'true'
        )
        if enable_evaluation:
            self.train_size = 1.0
            self.X_train = self.X
            self.y_train = self.y

            self.log(
                'No evaluation mode: using all data for training and skipping '
                'test split.'
            )

            return

        super().split()
        n_samples = self.X.shape[0]
        indices = np.arange(n_samples)

        if self.context.get('split_shuffle', True):
            np.random.seed(self.seed)
            np.random.shuffle(indices)

        test_size = 1 - self.train_size
        num_test_samples = max(2, int(n_samples * test_size))
        train_end = n_samples - num_test_samples

        self.X_train = self.X[indices[:train_end]]
        self.y_train = self.y[indices[:train_end]]
        self.X_test = self.X[indices[train_end:]]
        self.y_test = self.y[indices[train_end:]]

        m_train = self.X_train.shape[0] if self.X_train is not None else 0
        m_test = self.X_test.shape[0] if self.X_test is not None else 0
        self.log(
            f'Split data into {m_train} training samples '
            f'and {m_test} test samples.'
        )

    @override
    def train(self) -> None:
        """Validates that training data is ready before model training.

        Subclasses should override or extend this method to fit a regression
        pipeline and assign it to ``self.pipeline`` after calling
        ``super().train()``.

        Raises:
            ValueError: If the training split has not been created yet.
        """

        self.get_training_set()
        super().train()

    @override
    def evaluate(self) -> None:
        """Evaluates the pipeline on the test split and stores metrics.

        This implementation predicts on ``X_test`` and stores a
        ``RegressionMetrics`` instance in ``self.metrics`` containing MAE, MSE,
        R-squared, and MAPE.

        Raises:
            ValueError: If the test split or trained pipeline is unavailable.
        """

        X_test, y_test = self.get_test_set()
        model = self.get_model()
        super().evaluate()

        y_pred = model.predict(X_test)
        mae = mean_absolute_error(y_test, y_pred)
        rmse = mean_squared_error(y_test, y_pred) ** 0.5
        r2 = r2_score(y_test, y_pred)
        mape = mean_absolute_percentage_error(y_test, y_pred)

        self.context['metrics'] = RegressionMetrics(
            mae=mae, rmse=rmse, r2=r2, mape=mape
        )

    @override
    def get_metrics(self) -> RegressionMetrics:
        """Returns regression metrics with a concrete return type.

        Raises:
            ValueError: If metrics are not available in the context.
        """

        return cast(RegressionMetrics, super().get_metrics())

    def get_training_set(self) -> tuple[np.ndarray, np.ndarray]:
        """Returns the training split as ``(X_train, y_train)``.

        Raises:
            ValueError: If the training split has not been created yet.
        """

        if self.X_train is None or self.y_train is None:
            raise ValueError('Training data not set up. Call split() first.')

        return self.X_train, self.y_train

    def get_test_set(self) -> tuple[np.ndarray, np.ndarray]:
        """Returns the test split as ``(X_test, y_test)``.

        Raises:
            ValueError: If the test split has not been created yet.
        """

        if self.X_test is None or self.y_test is None:
            raise ValueError('Test data not set up. Call split() first.')

        return self.X_test, self.y_test

    # def plot(self) -> Plotter | None:
    #     """Returns a RegressionPlotter instance for visualizing the regression
    #     results.
    #     """
    #
    #     plotter_factory: Type[RegressionPlotter] | None = self.context.get(
    #         'plotter_factory'
    #     )
    #     if plotter_factory is None:
    #         return None
    #
    #     schema: DatasetSchema | None = self.context.get('schema')
    #     if schema is None:
    #         raise ValueError(
    #             'Dataset schema is required in context to create plotter.'
    #         )
    #
    #     plot_title: str = self.context.get('plot_title', self.name)
    #
    #     X_train, y_train = self.get_training_set()
    #
    #     if self.X_test is None or self.y_test is None:
    #         X_test, y_test = None, None
    #     else:
    #         X_test, y_test = self.get_test_set()
    #
    #     X_pred: np.ndarray | None = None
    #     if self.context.get('x-pred') is not None:
    #         X_pred_nums = self.context['x-pred'].split(',')
    #         X_pred = np.array(X_pred_nums, dtype=float).reshape(-1, 1)
    #
    #     plotter: Plotter = plotter_factory(
    #         schema=schema,
    #         title=plot_title,
    #         model=self.pipeline.predict,
    #         X_train=X_train,
    #         y_train=y_train,
    #         X_test=X_test,
    #         y_test=y_test,
    #         X_pred=X_pred,
    #         parameters=self.parameters,
    #         hyperparameters=self.hyperparameters,
    #     )
    #
    #     plotter.plot()
    #     return plotter
