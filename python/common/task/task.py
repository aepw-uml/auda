from abc import ABC, abstractmethod


class Task(ABC):
    @abstractmethod
    def run(self, **context) -> None:
        """Runs this task.

        Subclasses should implement this method to perform the specific actions
        of the task.
        """

        pass
