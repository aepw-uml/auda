from typing import Any, List

from pydantic import BaseModel


class DbISName:
    # Each statement should be separated by a semicolon.
    SQL_STATEMENTS = 'sql_statements'


class DbOSName:
    SQL_RESULTS = 'sql_results'


DB_KIND = 'db'


class SqlResults(BaseModel):
    """
    Model for SQL query results.

    Attributes:
        sql: The SQL query that was executed.
        column_names: The names of the columns in the result set.
        results: The rows returned by the SQL query, where each row is a list of values.
    """

    sql: str
    column_names: List[str]
    results: List[List[Any]]
