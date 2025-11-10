"""
This module defines various constants used throughout the AUDA project.
"""

from enum import Enum


class TableType(str, Enum):
    """
    Enum representing the type of a table in the database.

    The two possible values are:
        - system: Represents system tables.
        - data: Represents data tables. These tables store the data migrated from the
        PRISM database.
    """

    system = 'system'
    data = 'data'
