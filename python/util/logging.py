import logging

from util.env import env


def setup_logging() -> None:
    """Sets up the logging configuration for the project.

    The logging level is determined based on the current environment:
        - DEBUG level for DEVELOPMENT, TESTING, and STAGING environments.
        - INFO level for PRODUCTION environment.
    """

    level: int = logging.WARNING
    match env.environment.upper():
        case 'PRODUCTION':
            level = logging.DEBUG

    format: str = '%(asctime)s [%(levelname)s] %(name)s | %(message)s'
    match env.environment.upper():
        case 'PRODUCTION':
            format = '%(message)s'

    logging.basicConfig(format=format, level=level)


def get_logger(name: str) -> logging.Logger:
    """Returns a logger instance with the specified name."""

    return logging.getLogger(name)
