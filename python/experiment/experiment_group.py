from abc import abstractmethod

from dataset.dataset import Dataset
from experiment.experiment import Experiment
from util import logging


class ExperimentGroup:
    def __init__(self, name: str) -> None:
        self.name: str = name
        self.experiments: list[Experiment] = []
        self.context: dict[str, str] = {}
        self.logger = logging.get_logger(self.name)

    def set_context(self, **kwargs: str) -> None:
        self.context.update(kwargs)

    def add_experiment(self, experiment: Experiment) -> None:
        self.experiments.append(experiment)

    def run(self, dataset: Dataset) -> None:
        self.check_dataset(dataset)

    @abstractmethod
    def check_dataset(self, dataset: Dataset) -> None:
        pass
