from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, cast, override

import numpy as np
from sklearn.metrics import (
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_squared_error,
    r2_score,
)
from sklearn.pipeline import Pipeline


class Experiment(ABC):
    """Abstract base class for experiments.

    Subclasses are responsible for implementing the full experiment lifecycle:
    loading data in ``setup()``, creating dataset splits in ``split()``,
    fitting a pipeline in ``train()``, optionally tuning hyperparameters in
    ``tune()``, and computing metrics in ``evaluate()``.
    """

    def __init__(
        self,
        name: str,
        description: str,
        seed: int = 42,
        train_rate: float = 0.8,
        val_rate: float = 0.0,
    ) -> None:
        """Initializes an experiment.

        Args:
            name: Name of the experiment.
            description: Description of the experiment.
            seed: Random seed for reproducibility.
            train_rate: Proportion of data to use for training.
            val_rate: Proportion of data to use for validation.
        """

        self.name: str = name
        self.description: str = description
        self.seed: int = seed
        self.train_rate: float = train_rate
        self.val_rate: float = val_rate
        self.test_rate: float = 1.0 - train_rate - val_rate
        self.context: dict[str, Any] = {}
        self.pipeline: Pipeline | None = None
        self.metrics: Any = None

    @abstractmethod
    def setup(self, **kwargs) -> None:
        """Loads or assigns the inputs required by the experiment.

        Implementations should validate their inputs and store all state needed
        by later lifecycle methods, such as raw features, labels, or
        task-specific configuration. Keyword arguments passed to this method are
        merged into ``self.context`` so they remain available throughout the
        experiment lifecycle.

        This method must be called before ``split()``, ``train()``,
        ``tune()``, or ``evaluate()``.

        Args:
            **kwargs: Arbitrary experiment metadata or configuration to persist
                in ``self.context``. Values with duplicate keys overwrite
                earlier entries.
        """

        self.context = {**self.context, **kwargs}

    def run(self) -> None:
        """Runs the experiment from start to finish.

        This method first calls ``split()`` and ``train()``. Then, if
        ``val_rate > 0.0``, it calls ``tune()`` and ``train()`` again so the
        pipeline can be refit using the tuned configuration. Finally, it calls
        ``evaluate()`` to score the trained pipeline on the test split.
        """

        self.split()
        self.train()

        if self.val_rate > 0.0:
            self.tune()
            self.train()

        self.evaluate()

    @abstractmethod
    def split(self) -> None:
        """Splits the data into training, validation, and test sets.

        Subclasses should partition the data prepared in ``setup()`` according
        to ``train_rate``, ``val_rate``, and ``test_rate``. Implementations are
        expected to store the resulting splits on the instance so that
        ``train()``, ``tune()``, and ``evaluate()`` can consume them.

        The training split should be reserved for model fitting, the validation
        split for hyperparameter tuning when applicable, and the test split for
        final evaluation.
        """

        pass

    @abstractmethod
    def train(self) -> None:
        """Fits the experiment pipeline on the training split.

        Subclasses should create or update ``self.pipeline`` and train it using
        the data produced by ``split()``. When ``run()`` invokes ``train()`` a
        second time after ``tune()``, implementations should retrain the
        pipeline using the tuned configuration.
        """

        pass

    @abstractmethod
    def evaluate(self) -> None:
        """Evaluates the trained pipeline and stores the resulting metrics.

        Subclasses should use the held-out test split to generate predictions,
        compute task-appropriate metrics, and store the result in
        ``self.metrics`` so it can be returned by ``get_metrics()``.
        """

        pass

    @abstractmethod
    def tune(self) -> None:
        """Tunes hyperparameters using the validation split.

        Subclasses should use the validation data created by ``split()`` to
        search for improved model settings. Any tuned parameters should be
        stored in instance state so the next call to ``train()`` can rebuild or
        refit ``self.pipeline`` with the selected configuration.
        """

        pass

    def get_metrics(self) -> Any:
        """Returns the metrics computed during evaluation.

        Returns:
            The metrics object produced by ``evaluate()``.

        Raises:
            ValueError: If metrics have not been calculated yet.
        """

        if self.metrics is None:
            raise ValueError('Metrics not calculated. Call evaluate() first.')

        return self.metrics


@dataclass(frozen=True)
class RegressionMetrics:
    """Container for regression evaluation metrics.

    Attributes:
        mae: Mean Absolute Error.
        mse: Mean Squared Error.
        r2: R-squared score.
        mape: Mean Absolute Percentage Error.
    """

    mae: float
    mse: float
    r2: float
    mape: float

    def __repr__(self) -> str:
        """Returns a compact string representation of the metric values."""

        return (
            f'Metrics(mae={self.mae:.4f}, mse={self.mse:.4f}, '
            f'r2={self.r2:.4f}, mape={self.mape:.4f})'
        )


class RegressionExperiment(Experiment):
    """Base implementation for regression experiments.

    This class manages regression dataset storage, randomized splitting, and
    metric computation. Concrete subclasses are still expected to implement the
    model-specific training and tuning behavior by extending or overriding
    ``train()`` and ``tune()``.
    """

    def __init__(
        self,
        name: str,
        description: str,
        seed: int = 42,
        train_rate: float = 0.8,
        val_rate: float = 0.0,
    ) -> None:
        """Initializes a regression experiment.

        The instance starts without any dataset assigned. ``setup()`` must be
        called before the experiment can be split, trained, tuned, or
        evaluated.
        """

        super().__init__(name, description, seed, train_rate, val_rate)

        self.X: np.ndarray | None = None
        self.y: np.ndarray | None = None
        self.X_train: np.ndarray | None = None
        self.y_train: np.ndarray | None = None
        self.X_val: np.ndarray | None = None
        self.y_val: np.ndarray | None = None
        self.X_test: np.ndarray | None = None
        self.y_test: np.ndarray | None = None

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

    @override
    def split(self) -> None:
        """Randomly splits the dataset into train, validation, and test sets.

        The shuffled splits are written to ``X_train``, ``y_train``, ``X_val``,
        ``y_val``, ``X_test``, and ``y_test`` for downstream use by
        ``train()``, ``tune()``, and ``evaluate()``.

        Raises:
            ValueError: If the dataset has not been provided via ``setup()``.
        """

        if self.X is None or self.y is None:
            raise ValueError('Data not set up. Call setup() first.')

        n_samples = self.X.shape[0]
        indices = np.arange(n_samples)
        np.random.seed(self.seed)
        np.random.shuffle(indices)

        train_end = int(self.train_rate * n_samples)
        val_end = train_end + int(self.val_rate * n_samples)

        self.X_train = self.X[indices[:train_end]]
        self.y_train = self.y[indices[:train_end]]
        self.X_val = self.X[indices[train_end:val_end]]
        self.y_val = self.y[indices[train_end:val_end]]
        self.X_test = self.X[indices[val_end:]]
        self.y_test = self.y[indices[val_end:]]

    def train(self) -> None:
        """Validates that training data is ready before model training.

        Subclasses should override or extend this method to fit a regression
        pipeline and assign it to ``self.pipeline``.

        Raises:
            ValueError: If the training split has not been created yet.
        """

        if self.X_train is None or self.y_train is None:
            raise ValueError('Training data not set up. Call split() first.')

    def evaluate(self) -> None:
        """Evaluates the pipeline on the test split and stores metrics.

        This implementation predicts on ``X_test`` and stores a
        ``RegressionMetrics`` instance in ``self.metrics`` containing MAE, MSE,
        R-squared, and MAPE.

        Raises:
            ValueError: If the test split or trained pipeline is unavailable.
        """

        if self.X_test is None or self.y_test is None:
            raise ValueError('Test data not set up. Call split() first.')

        if self.pipeline is None:
            raise ValueError('Pipeline not set up. Call train() first.')

        y_pred = self.pipeline.predict(self.X_test)
        mae = mean_absolute_error(self.y_test, y_pred)
        mse = mean_squared_error(self.y_test, y_pred)
        r2 = r2_score(self.y_test, y_pred)
        mape = mean_absolute_percentage_error(self.y_test, y_pred)
        self.metrics = RegressionMetrics(mae=mae, mse=mse, r2=r2, mape=mape)

    def tune(self) -> None:
        """Validates that validation data is ready for hyperparameter tuning.

        Subclasses should override or extend this method to search regression
        hyperparameters using the validation split and persist the chosen
        settings for the next call to ``train()``.

        Raises:
            ValueError: If the validation split has not been created yet.
        """

        if self.X_val is None or self.y_val is None:
            raise ValueError('Validation data not set up. Call split() first.')

    def get_metrics(self) -> RegressionMetrics:
        """Returns regression metrics with a concrete return type."""

        return cast(RegressionMetrics, super().get_metrics())
