import time
from abc import ABC
from typing import Any, Type, cast, override

import numpy as np
from common.hyperparameters import get_hyperparameters_str
from common.metrics import RegressionMetrics
from dataset.dataset import DatasetSchema
from sklearn.metrics import (
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_squared_error,
    r2_score,
)
from sklearn.pipeline import Pipeline
from step.plot.plotter import Plotter, RegressionPlotter
from util.logging import get_logger


class Experiment(ABC):
    """Abstract base class for experiments.

    Subclasses implement the experiment lifecycle: load inputs in
    ``setup()``, create splits in ``split()``, optionally choose
    hyperparameters in ``tune()``, fit a pipeline in ``train()``, and compute
    metrics in ``evaluate()``.

    Attributes:
        name: Human-readable experiment name used in logs.
        description: Short description of the experiment's purpose.
        seed: Random seed used for reproducible operations.
        train_rate: Fraction of samples assigned to the training split.
        val_rate: Fraction of samples assigned to the validation split.
        test_rate: Fraction of samples assigned to the test split.
        context: Shared experiment configuration and runtime metadata.
        logger: Logger used to report lifecycle progress and results.
        pipeline: Trained scikit-learn pipeline produced during training.
        parameters: General experiment parameters persisted on the instance.
        hyperparameters: Tuned model settings selected before training.
        metrics: Evaluation result produced by ``evaluate()``.
    """

    def __init__(
        self,
        name: str,
        description: str,
        seed: int = 42,
        train_size: float = 0.8,
    ) -> None:
        """Initializes an experiment.

        Args:
            name: Name of the experiment.
            description: Description of the experiment.
            seed: Random seed for reproducibility.
            train_rate: Proportion of data to use for training.
        """

        self.name: str = name
        self.description: str = description
        self.seed: int = seed
        self.train_rate: float = train_size

        self.test_rate: float = 1.0 - train_size
        self.context: dict[str, Any] = {}
        self.logger = get_logger(name)

        self.pipeline: Any = None
        self.parameters: dict[str, Any] = {}
        self.hyperparameters: dict[str, Any] = {}
        self.metrics: Any = None

    def set_context(self, **kwargs) -> None:
        """Updates the experiment context with additional key-value pairs.

        This method provides a convenient way to persist arbitrary metadata or
        configuration throughout the experiment lifecycle. Values with duplicate
        keys overwrite earlier entries.

        Args:
            **kwargs: Key-value pairs to merge into the experiment context.
        """

        self.context = {**self.context, **kwargs}

    def set_hyperparameters(self, replace: bool = False, **kwargs) -> None:
        """Updates the experiment's hyperparameters with additional key-value
        pairs.

        Args:
            replace: If True, the provided hyperparameters replace any existing
                values. If False, the provided hyperparameters are merged with
                existing values, with precedence given to the new entries.
            **kwargs: Hyperparameter names and values to merge into the context
                and update on the instance.
        """

        self.hyperparameters = (
            {**self.hyperparameters, **kwargs} if not replace else kwargs
        )
        self.context['hyperparameters'] = self.hyperparameters

    def log(self, message: str) -> None:
        """Logs a message with the experiment's logger.

        Args:
            message: The message to log.
        """

        if self.logger is not None and self.context.get('enable_logging', True):
            self.logger.info(message)

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

        # Set up Hyperparameters.
        if 'hyperparameters' in self.context:
            self.set_hyperparameters(
                replace=False, **self.context['hyperparameters']
            )
        else:
            self.set_hyperparameters(replace=True)

    def run(self) -> None:
        """Runs the experiment from start to finish.

        The workflow is ``split()``, optional ``tune()``, ``train()``, then
        ``evaluate()``. Tuning only runs when ``val_rate > 0.0``.
        """

        self.split()
        self.tune()
        self.train()
        self.evaluate()

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

        self.log(
            f'Splitting data with train_rate={self.train_rate}, '
            f'test_rate={self.test_rate}'
        )

    def train(self) -> None:
        """Fits the experiment pipeline on the training split.

        Subclasses should create or update ``self.pipeline`` and fit it using
        the training data produced by ``split()``. If ``tune()`` stores
        selected hyperparameters, ``train()`` should use them.
        """

        hyperparameters_str: str = get_hyperparameters_str(self.hyperparameters)
        self.log(f'Training ({hyperparameters_str})...')

    def evaluate(self) -> None:
        """Evaluates the trained pipeline and stores the resulting metrics.

        Subclasses should use the held-out test split to generate predictions,
        compute task-appropriate metrics, and store the result in
        ``self.metrics`` so it can be returned by ``get_metrics()``.
        """

        self.log('Evaluating...')

    def tune(self) -> None:
        """Tunes hyperparameters using the validation split.

        Subclasses should use the validation data created by ``split()`` to
        search for improved model settings. Any tuned parameters should be
        stored in instance state so ``train()`` can build the final pipeline
        with the selected configuration.
        """

        pass

    def get_pipeline(self) -> Pipeline:
        """Returns the trained pipeline.

        Returns:
            The trained scikit-learn Pipeline instance.

        Raises:
            ValueError: If the pipeline has not been trained yet.
        """

        if self.pipeline is None:
            raise ValueError('Pipeline not set up. Call train() first.')

        return self.pipeline

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

    def finish(self) -> Any:
        """Performs any finalization steps after the experiment has run.

        The base implementation only logs completion. Subclasses can override
        this hook to persist artifacts or emit summaries.
        """

        self.log('Experiment finished.')

    def get_training_set(self) -> Any:
        """Returns the training split prepared by ``split()``.

        The base class does not define a concrete dataset representation, so
        subclasses are expected to override this method with a task-specific
        return type and validation.
        """

        pass

    def get_test_set(self) -> Any:
        """Returns the test split prepared by ``split()``.

        The base class leaves the concrete dataset representation to
        subclasses, so implementations should override this method with the
        appropriate return type and readiness checks.
        """

        pass

    def plot(self) -> Plotter | None:
        """Returns a Plotter instance for visualizing the experiment results.

        The base class can return a Plotter instance if the experiment is
        designed to produce visualizations, but subclasses can override this
        method to return None if plotting is not applicable.
        """

        return None

    def timer_start(self, name: str) -> None:
        """Starts a timer with the given name for measuring execution time.

        Args:
            name: The name of the timer to start.
        """

        self.context[f'{name}_start'] = time.perf_counter()

    def timer_stop(self, name: str) -> None:
        """Stops the timer with the given name and logs the elapsed time.

        Args:
            name: The name of the timer to stop.
        """

        start_time = self.context.get(f'{name}_start')
        if start_time is None:
            self.log(f'Timer "{name}" was not started.')
            return

        end_time = time.perf_counter()
        elapsed_ms = (end_time - start_time) * 1000
        self.log(f'Timer "{name}" stopped. Elapsed time: {elapsed_ms} seconds.')

        self.context.pop(f'{name}_start', None)
        self.context[f'{name}_elapsed_ms'] = elapsed_ms


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
        seed: int = 42,
        train_rate: float = 0.8,
    ) -> None:
        """Initializes a regression experiment.

        The instance starts without any dataset assigned, so ``setup()`` must
        run before the experiment can be split, trained, tuned, or evaluated.
        """

        super().__init__(name, description, seed, train_rate)
        self.metrics: RegressionMetrics | None = None

        self.X: np.ndarray | None = None
        self.y: np.ndarray | None = None
        self.X_train: np.ndarray | None = None
        self.y_train: np.ndarray | None = None
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

        if self.context.get('no-evaluation', 'False').lower() == 'true':
            self.test_rate = 0.0
            self.train_rate = 1.0

            self.X_train = self.X
            self.y_train = self.y

            self.log(
                'No evaluation mode: using all data for training and skipping '
                'test split.'
            )

            return

        n_samples = self.X.shape[0]
        indices = np.arange(n_samples)

        if self.context.get('split_shuffle', True):
            np.random.seed(self.seed)
            np.random.shuffle(indices)

        num_test_samples = max(2, int(n_samples * self.test_rate))
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

        if self.context.get('no-evaluation', 'False').lower() == 'true':
            self.metrics = RegressionMetrics()
            return

        self.get_test_set()
        pipeline = self.get_pipeline()
        super().evaluate()

        y_pred = pipeline.predict(self.X_test)
        mae = mean_absolute_error(self.y_test, y_pred)
        rmse = mean_squared_error(self.y_test, y_pred) ** 0.5
        r2 = r2_score(self.y_test, y_pred)
        mape = mean_absolute_percentage_error(self.y_test, y_pred)
        self.metrics = RegressionMetrics(mae=mae, rmse=rmse, r2=r2, mape=mape)

    @override
    def get_metrics(self) -> RegressionMetrics:
        """Returns regression metrics with a concrete return type."""

        return cast(RegressionMetrics, super().get_metrics())

    @override
    def get_training_set(self) -> tuple[np.ndarray, np.ndarray]:
        """Returns the training split as ``(X_train, y_train)``.

        Raises:
            ValueError: If the training split has not been created yet.
        """

        if self.X_train is None or self.y_train is None:
            raise ValueError('Training data not set up. Call split() first.')

        return self.X_train, self.y_train

    @override
    def get_test_set(self) -> tuple[np.ndarray, np.ndarray]:
        """Returns the test split as ``(X_test, y_test)``.

        Raises:
            ValueError: If the test split has not been created yet.
        """

        if self.X_test is None or self.y_test is None:
            raise ValueError('Test data not set up. Call split() first.')

        return self.X_test, self.y_test

    def plot(self) -> Plotter | None:
        """Returns a RegressionPlotter instance for visualizing the regression
        results.
        """

        plotter_factory: Type[RegressionPlotter] | None = self.context.get(
            'plotter_factory'
        )
        if plotter_factory is None:
            return None

        schema: DatasetSchema | None = self.context.get('schema')
        if schema is None:
            raise ValueError(
                'Dataset schema is required in context to create plotter.'
            )

        plot_title: str = self.context.get('plot_title', self.name)

        X_train, y_train = self.get_training_set()

        if self.X_test is None or self.y_test is None:
            X_test, y_test = None, None
        else:
            X_test, y_test = self.get_test_set()

        X_pred: np.ndarray | None = None
        if self.context.get('x-pred') is not None:
            X_pred_nums = self.context['x-pred'].split(',')
            X_pred = np.array(X_pred_nums, dtype=float).reshape(-1, 1)

        plotter: Plotter = plotter_factory(
            schema=schema,
            title=plot_title,
            model=self.pipeline.predict,
            X_train=X_train,
            y_train=y_train,
            X_test=X_test,
            y_test=y_test,
            X_pred=X_pred,
            parameters=self.parameters,
            hyperparameters=self.hyperparameters,
        )

        plotter.plot()
        return plotter
