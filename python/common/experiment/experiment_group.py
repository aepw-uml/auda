from abc import abstractmethod

from common.dataset import Dataset
from common.experiment.experiment import Experiment
from common.vars import Vars
from util.logging import get_logger


class ExperimentGroup:
    def __init__(self, name: str) -> None:
        self.name: str = name
        self.experiments: list[Experiment] = []
        self.context: Vars = {}
        self.logger = get_logger(self.name)

    def set_context(self, **kwargs: str) -> None:
        self.context.update(kwargs)

    def add_experiment(self, experiment: Experiment) -> None:
        self.experiments.append(experiment)

    def run(self, dataset: Dataset) -> None:
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
