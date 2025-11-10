from os import getenv


class Env:
    """
    Loads and validates required environment variables for the application.

    This class reads predefined environment variable names (as annotated attributes)
    from a `.env` file using `python-dotenv` and sets them as instance attributes.
    If any required variable is missing, it raises a ValueError with a helpful message.

    Attributes are grouped into categories such as server settings, database settings,
    JWT configuration, and email service configuration.
    """

    ENVIRONMENT: str

    # Database settings
    AUDA_DB_HOST: str
    AUDA_DB_PORT: str
    AUDA_DB_DBNAME: str
    AUDA_DB_USERNAME: str
    AUDA_DB_PASSWORD: str

    PRISM_DB_HOST: str
    PRISM_DB_PORT: str
    PRISM_DB_DBNAME: str
    PRISM_DB_USERNAME: str
    PRISM_DB_PASSWORD: str

    # OpenAI (ChatGPT) settings
    OPENAI_API_KEY: str

    def __init__(self, dotenv_path: str = '.env'):
        """
        Initializes the Env class and loads environment variables from a specified
        environment file.

        Args:
            dotenv_path: The path to the .env file. Defaults to '.env'.
        """
        from dotenv import load_dotenv

        # Load environment variables from the .env file
        load_dotenv(dotenv_path=dotenv_path)

        for attr_name in self.__annotations__:
            setattr(self, attr_name, self.__get(attr_name))

    @staticmethod
    def __get(name: str) -> str:
        """
        Gets environment variable value or raise an error if not set.

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
                'Please define it in the .env file.'
            )

        return value
