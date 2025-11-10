from abc import ABC, abstractmethod


class LargeLanguageModel(ABC):
    def __init__(self, name: str):
        """
        Initializes the large language model.

        Args:
            name: The name of the large language model.

        Attributes:
            name: The name of the large language model.
            developer_prompt: The developer prompt to be used with the model.
        """
        self.name: str = name
        self.developer_prompt: str = ''

    @abstractmethod
    def ask(self, user_prompt: str, developer_prompt: str | None = None) -> str:
        """
        Sends a prompt to the large language model and returns the response.

        Args:
            developer_prompt: The prompt from the developer or system.
            user_prompt: The prompt from the end user.

        Returns:
            The response from the large language model as a string.
        """
        pass
