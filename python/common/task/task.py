from abc import ABC, abstractmethod

from common.dataset import Dataset
from common.experiment.experiment import Experiment
from common.vars import Vars
from util.logging import get_logger


class Task(ABC):
    """Task groups related experiments for one problem definition.

    Attributes:
        name: Name of the task.
        experiments: Experiments that belong to this task.
        context: A dictionary for storing arbitrary key-value pairs related to
            the task's state and configuration.
        logger: A logging object for recording messages and events during the
            task's execution.
    """

    def __init__(self, name: str) -> None:
        """Initializes a task.

        Args:
            name: The name of the task.
        """

        self.name: str = name
        self.experiments: list[Experiment] = []
        self.context: Vars = {}
        self.logger = get_logger(self.name)

    def set_context(self, **kwargs: str) -> None:
        """Updates the task context with the provided key-value pairs.

        Args:
            **kwargs: Arbitrary key-value pairs to add to the task context.
        """

        self.context.update(kwargs)

    def add(self, experiment: Experiment) -> None:
        """Adds an experiment to the task.

        Args:
            experiment: The experiment to add to the task.
        """

        experiment.set_context(**self.context)
        self.experiments.append(experiment)

    def run(self, dataset: Dataset) -> None:
        """Runs all experiments in the task on the provided dataset.

        Args:
            dataset: The dataset to use for running the task.
        """

        self.check_dataset(dataset)

    @abstractmethod
    def check_dataset(self, dataset: Dataset) -> None:
        """Checks dataset compatibility for this task.

        Subclasses should implement this method to ensure that the dataset
        meets the requirements of the experiments in this task.

        Args:
            dataset: The dataset to check.
        """

        pass
