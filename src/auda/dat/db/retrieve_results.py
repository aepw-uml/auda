from typing import List, override

from sqlalchemy import text

from auda.core import DatabaseName, project
from auda.utils.pipeline import IOSpec, Task, task

from .__common import DB_KIND, DbISName, DbOSName, SqlResults


@task(
    id='RETRIEVE-RESULTS',
    kind=DB_KIND,
    description='Executes SQL queries and retrieves their results from the database.',
    input_specs={
        DbISName.SQL_STATEMENTS: IOSpec(dtype=str),
    },
    output_specs={
        DbOSName.SQL_RESULTS: IOSpec(dtype=List[SqlResults]),
    },
)
class RetrieveResults(Task):
    @override
    def run(self) -> None:
        sql_statements: List[str] = self.get_input(DbISName.SQL_STATEMENTS)
        results: List[SqlResults] = [self.get_sql_result(sql) for sql in sql_statements]

        self.set_output(DbOSName.SQL_RESULTS, results)

    def get_sql_result(self, sql: str) -> SqlResults:
        """
        Execute the given SQL statement and return the results.

        Args:
            sql: The SQL statement to execute.

        Returns:
            The results of the SQL execution.
        """
        auda_database = project.database_manager.get(DatabaseName.AUDA)

        with auda_database.engine.connect() as connection:
            result = connection.execute(text(sql))

            return SqlResults(
                sql=sql,
                column_names=list(map(str, result.keys())),
                results=[list(row) for row in result.fetchall()],
            )
