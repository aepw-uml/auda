from logging import DEBUG, INFO, WARNING, Logger, basicConfig, getLogger
from pathlib import Path

from .constants import DatabaseName, Environment
from .database import Database, DatabaseManager
from .env import Env
from .singleton_registry import SingletonRegistry
from .template_manager import TemplateManager


class Project:
    def __init__(self):
        # Cache directory storing cached files
        self.cache_dir: Path = Path.cwd() / 'cache'

        # Results directory storing output files
        self.results_dir: Path = Path.cwd() / 'results'

        # Template directory storing template files
        self.templates_dir: Path = Path.cwd() / 'templates'

        # Environment configuration for the project
        self.env = Env()

        # Singleton registry
        self.singleton = SingletonRegistry()

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

        # Template manager
        self.template = TemplateManager(self.templates_dir)

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

        format: str = '%(asctime)s [%(levelname)s] %(name)s | %(message)s'

        level: int = WARNING
        match self.env.ENVIRONMENT.lower():
            case (
                Environment.DEVELOPMENT
                | Environment.TESTING
                | Environment.STAGING
            ):
                level = DEBUG
            case Environment.PRODUCTION:
                level = INFO

        basicConfig(format=format, level=level)
