from logging import DEBUG, INFO, Logger, basicConfig, getLogger
from pathlib import Path
from string import Template

from diskcache import Cache

from .constants import Environment
from .database import Database, DatabaseManager, DatabaseName
from .env import Env
from .singleton_registry import SingletonRegistry


class Project:
    def __init__(self):
        # The current working directory
        self.cwd: Path = Path.cwd()

        # The cache directory storing cached files
        self.cache_dir: Path = self.cwd / 'cache'

        # The results directory storing output files
        self.results_dir: Path = self.cwd / 'results'

        # The template directory storing template files
        self.templates_dir: Path = self.cwd / 'templates'

        # The environment configuration for the project
        self.env = Env()

        # Database manager and database credentials
        self.database_manager = DatabaseManager()
        self.database_manager.databases[DatabaseName.AUDA] = Database(
            host=self.env.AUDA_DB_HOST,
            port=int(self.env.AUDA_DB_PORT),
            dbname=self.env.AUDA_DB_DBNAME,
            username=self.env.AUDA_DB_USERNAME,
            password=self.env.AUDA_DB_PASSWORD,
        )
        self.database_manager.databases[DatabaseName.PRISM] = Database(
            host=self.env.PRISM_DB_HOST,
            port=int(self.env.PRISM_DB_PORT),
            dbname=self.env.PRISM_DB_DBNAME,
            username=self.env.PRISM_DB_USERNAME,
            password=self.env.PRISM_DB_PASSWORD,
        )

        # File Cache
        self.cache = Cache(self.cache_dir)

        # Singleton registry
        self.singleton_registry = SingletonRegistry()

    def get_logger(self, name: str) -> Logger:
        """
        Returns a logger instance with the specified name.
        """
        return getLogger(name)

    def setup_logging(self) -> None:
        """
        Sets up the logging configuration for the project.

        The logging level is determined based on the current environment:
            - DEBUG level for DEVELOPMENT, TESTING, and STAGING environments.
            - INFO level for PRODUCTION environment.
        """
        level: int = DEBUG
        match self.env.ENVIRONMENT.lower():
            case Environment.DEVELOPMENT | Environment.TESTING | Environment.STAGING:
                level = DEBUG
            case Environment.PRODUCTION:
                level = INFO

        basicConfig(
            level=level,
            format='%(asctime)s [%(levelname)s] %(name)s | %(message)s',
        )

    def get_template(self, template_name: str) -> Template:
        """
        Reads and returns the content of a template file.

        Args:
            template_name: The name of the template file.

        Returns:
            The content of the template file as a string.
        """
        template_path = self.templates_dir / template_name
        with open(template_path, 'r', encoding='utf-8') as file:
            content = file.read()
            # Remove all the lines that start with '#'
            content = '\n'.join(
                line
                for line in content.splitlines()
                if not line.strip().startswith('#')
            )

            return Template(content)
