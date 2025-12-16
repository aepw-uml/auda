from enum import Enum


class Environment:
    """Enumeration of supported environments."""

    DEVELOPMENT = 'development'
    TESTING = 'testing'
    STAGING = 'staging'
    PRODUCTION = 'production'


class DatabaseName(str, Enum):
    """Enumeration of supported database names."""

    AUDA = 'auda'
    PRISM = 'prism'
