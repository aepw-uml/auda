import logging

from util.env import env


def setup_logging() -> None:
    """Sets up the logging configuration for the project.

    The logging level is determined based on the current environment:
        - DEBUG level for DEVELOPMENT, TESTING, and STAGING environments.
        - INFO level for PRODUCTION environment.
    """

    level: int = logging.INFO
    match env.environment.upper():
        case 'PRODUCTION':
            level = logging.WARNING

    logging.basicConfig(
        format='%(asctime)s [%(levelname)s] %(name)s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        level=level,
    )


# Indicates whether the logging configuration has been set up.
has_setup: bool = False


def get_logger(name: str) -> logging.Logger:
    """Returns a logger instance with the specified name."""

    if not has_setup:
        setup_logging()

    return logging.getLogger(name)
