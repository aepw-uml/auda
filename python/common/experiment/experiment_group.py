from abc import abstractmethod

from common.dataset import Dataset
from common.experiment.experiment import Experiment
from common.vars import Vars
from util.logging import get_logger


class ExperimentGroup:
    """ExperimentGroup is a class for grouping related experiments together.

    Attributes:
        name: A string representing the name of the experiment group.
        experiments: A list of Experiment instances that belong to this group.
        context: A dictionary for storing arbitrary key-value pairs related to
            the experiment group's state and configuration.
        logger: A logging object for recording messages and events during the
            experiment group's execution.
    """

    def __init__(self, name: str) -> None:
        """Initializes an ExperimentGroup instance.

        Args:
            name: The name of the experiment group.
        """

        self.name: str = name
        self.experiments: list[Experiment] = []
        self.context: Vars = {}
        self.logger = get_logger(self.name)

    def set_context(self, **kwargs: str) -> None:
        """Updates the context of the experiment group with the provided
        key-value pairs.

        Args:
            **kwargs: Arbitrary key-value pairs to add to the experiment group's
                context.
        """

        self.context.update(kwargs)

    def add(self, experiment: Experiment) -> None:
        """Adds an Experiment instance to the experiment group.

        Args:
            experiment: The Experiment instance to add to the group.
        """

        experiment.set_context(**self.context)
        self.experiments.append(experiment)

    def run(self, dataset: Dataset) -> None:
        """Runs all experiments in the group on the provided dataset.

        Args:
            dataset: The dataset to use for running the experiments.
        """

        self.check_dataset(dataset)

    @abstractmethod
    def check_dataset(self, dataset: Dataset) -> None:
        """Checks the dataset for compatibility with the experiments in this
        group.

        Subclasses should implement this method to ensure that the dataset meets
        the requirements of the experiments in this group.

        Args:
            dataset: The dataset to check.
        """

        pass
