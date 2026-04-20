from abc import ABC, abstractmethod


class Workflow(ABC):
    @abstractmethod
    def run(self, **context) -> None:
        """Runs this workflow.

        Subclasses should implement this method to perform the specific actions
        of the workflow.
        """

        pass
