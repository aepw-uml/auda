from os import getenv
from typing import Literal, cast

from dotenv import load_dotenv

Environment = Literal['DEVELOPMENT', 'TESTING', 'PRODUCTION']


class Env:
    """Loads and validates required environment variables for the application.

    This class uses the `python-dotenv` library to load environment variables
    from a specified `.env` file. It ensures that all required environment
    variables are set and provides easy access to them as class attributes.
    """

    def __init__(self, dotenv_path: str):
        """Initializes the Env class and loads environment variables from a
        specified environment file.

        Args:
            dotenv_path: The path to the .env file; defaults to '.env'.
        """

        load_dotenv(dotenv_path=dotenv_path)
        self.__path: str = dotenv_path

        self.environment: Environment = cast(
            Environment, self.__get('ENVIRONMENT')
        )
        self.dbUrl: str = self.__get('DB_URL')

    def __get(self, name: str) -> str:
        """Gets environment variable value or raise an error if not set.

        Args:
            name: The name of the environment variable to get.

        Returns:
            The value of the environment variable.

        Raises:
            ValueError: If the environment variable is not set.
        """

        value = getenv(name)

        if value is None:
            raise ValueError(
                f"Environment variable '{name}' is undefined. "
                f'Please define it in the ${self.__path} file.'
            )

        return value


env: Env = Env('.env')
