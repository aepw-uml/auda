import time
from abc import ABC
from typing import Any

from common.vars import Vars
from step.plot.plotter import Plotter
from util.logging import get_logger


class Experiment(ABC):
    """Experiment is a base class for defining machine learning experiments.

    Attributes:
        name: A string representing the name of the experiment.
        description: A string providing a brief description of the experiment.
        train_size: A float between 0 and 1 indicating the proportion of data
            to use for training (default is 0.9).
        seed: An integer used to seed random number generators for
            reproducibility.
        context: A dictionary for storing arbitrary key-value pairs related to
            the experiment's state and configuration.
        logger: A logging object for recording messages and events during the
            experiment.
        model: An object representing the machine learning model used in the
            experiment.
        hyperparameters: A dictionary for storing hyperparameters used in the
            experiment.
        parameters: A dictionary for storing parameters derived from the
            experiment's execution.

    ContextKeys:
        enable_logging: A boolean indicating whether logging is enabled; default
            to True.
    """

    def __init__(
        self,
        name: str,
        description: str,
        train_size: float = 0.8,
        seed: int = 471,
    ):
        """Initializes an Experiment instance.

        Args:
            name: The name of the experiment.
            description: A brief description of the experiment.
            train_size: The proportion of data to use for training.
            seed: An integer seed for reproducibility.
        """

        self.name = name
        self.description = description
        self.train_size = train_size
        self.seed = seed

        self.context: Vars = {}
        self.logger = get_logger(name)

        self.model: Any = None
        self.hyperparameters: Vars = {}
        self.parameters: Vars = {}

    def set_context(self, **kwargs) -> None:
        """Updates the experiment context with additional key-value pairs.

        This method provides a convenient way to persist arbitrary metadata or
        configuration throughout the experiment lifecycle. Values with duplicate
        keys overwrite earlier entries.

        Args:
            **kwargs: Key-value pairs to merge into the experiment context.
        """

        self.context = {**self.context, **kwargs}

    @staticmethod
    def parse_bool(value: object, default: bool = False) -> bool:
        """Parses a context value into a boolean.

        Args:
            value: Raw context value to interpret.
            default: Boolean value to use when ``value`` is ``None``.

        Returns:
            The parsed boolean value.

        Raises:
            ValueError: If a string value cannot be interpreted as boolean.
        """

        if value is None:
            return default

        if isinstance(value, bool):
            return value

        if isinstance(value, int):
            return value != 0

        normalized = str(value).strip().lower()
        if normalized in ('1', 'true', 'yes', 'y', 'on'):
            return True

        if normalized in ('0', 'false', 'no', 'n', 'off'):
            return False

        raise ValueError(
            f'Invalid boolean value: {value!r}. '
            'Expected one of 1/0, true/false, yes/no, or on/off.'
        )

    def get_context_bool(self, key: str, default: bool = False) -> bool:
        """Returns a boolean value from the experiment context.

        Args:
            key: Context key to retrieve.
            default: Boolean value to use when the key is absent.

        Returns:
            The parsed boolean value for the context key.
        """

        return self.parse_bool(self.context.get(key), default)

    def log(self, message: str) -> None:
        """Logs a message with the experiment's logger.

        This function logs the provided message at the INFO level, but only if
        logging is enabled in the experiment context.

        Args:
            message: The message to log.
        """

        enable_logging = self.get_context_bool('enable_logging', True)
        if self.logger is not None and enable_logging:
            self.logger.info(message)

    def setup(self, **kwargs) -> None:
        """Performs initial setup for the experiment.

        Subclasses should implement this method to prepare the dataset and any
        necessary configuration for the experiment. The provided keyword
        arguments can be used to pass in data or metadata, and they are also
        merged into the experiment context for later access.

        Args:
            **kwargs: Arbitrary keyword arguments for experiment setup.
        """

        self.set_context(**kwargs)

    def anomaly_detection(self) -> None:
        """Performs anomaly detection on the dataset.

        Subclasses should implement this method to identify and handle anomalies
        in the dataset, such as outliers or missing values. The results of
        anomaly detection can be stored in the experiment context for later use.
        """

        pass

    def split(self) -> None:
        """Splits the dataset into a training set and a test set based on the
        specified train size.

        Subclasses should partition the data prepared for the experiment
        according to the `train_size` attribute, ensuring that the split is
        reproducible using the provided `seed`.
        """

        self.log(
            f'Splitting data with train_rate={self.train_size:.2f}, '
            f'test_rate={1 - self.train_size:.2f}, seed={self.seed}.',
        )

    def tune(self) -> None:
        """Tunes hyperparameters for the experiment.

        Subclasses should implement this method to perform hyperparameter tuning
        using the training set. The tuned hyperparameters should be stored in
        the `hyperparameters` attribute for later use during training and
        evaluation.
        """

        pass

    def train(self) -> None:
        """Fits the experiment model on the training split.

        Subclasses should update ``self.model`` and fit it using the training
        data produced by ``split()``. If ``tune()`` stores selected
        hyperparameters, ``train()`` should use them.
        """

        self.log('Training model...')

    def evaluate(self) -> None:
        """Evaluates the trained model and stores the resulting metrics.

        Subclasses should use the held-out test split to generate predictions,
        compute task-appropriate metrics, and store the result in the "metrics"
        context key for later analysis.
        """

        self.log('Evaluating...')

    def finish(self) -> Any:
        """Performs any finalization steps after the experiment has run.

        The base implementation only logs completion. Subclasses can override
        this hook to persist artifacts or emit summaries.
        """

        self.log('Experiment finished.')

    def run(self) -> None:
        """Runs the experiment from start to finish.

        ``split()`` runs before ``anomaly_detection()`` so that any anomaly
        removal is fit on, and applied to, the training split alone. This keeps
        the held-out test set untouched and prevents test observations from
        influencing outlier detection.
        """

        self.split()
        self.anomaly_detection()
        self.tune()
        self.train()
        self.evaluate()
        self.finish()

    def get_model(self) -> Any:
        """Returns the trained regression model.

        Raises:
            ValueError: If the model is not trained or does not have a
                predict method.
        """

        if self.model is None:
            raise ValueError('Model not trained. Call train() first.')

        if not hasattr(self.model, 'predict'):
            raise ValueError(
                'Model does not have a predict method. Ensure it is a '
                'regression model.'
            )

        return self.model

    def get_metrics(self) -> Any:
        """Returns metrics with a concrete return type.

        Raises:
            ValueError: If metrics are not available in the context.
        """

        metrics = self.context.get('metrics')
        if metrics is None:
            raise ValueError('Metrics not available. Call evaluate() first.')

        return metrics

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
        self.log(
            f'Timer "{name}" stopped. Elapsed time: {elapsed_ms} milliseconds.'
        )

        self.context.pop(f'{name}_start', None)
        self.context[f'{name}_elapsed_ms'] = elapsed_ms
