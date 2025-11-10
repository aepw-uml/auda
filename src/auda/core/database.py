from contextlib import contextmanager
from enum import Enum
from typing import Dict, Generator

from sqlalchemy import create_engine
from sqlalchemy.dialects import postgresql
from sqlalchemy.orm import Query, Session, sessionmaker


class DatabaseName(str, Enum):
    AUDA = 'auda'
    PRISM = 'prism'


class Database:
    """
    Represents a database connection and session manager using SQLAlchemy.
    """

    def __init__(
        self,
        host: str,
        port: int,
        dbname: str,
        username: str,
        password: str,
        protocol: str = 'postgresql+psycopg2',
    ):
        """
        Initializes the database connection URL, engine, and session factory.

        Args:
            host: Database host address.
            port: Database port number.
            dbname: Database name.
            username: Username for authentication.
            password: Password for authentication.
            protocol: Database protocol and driver.
        """
        self.url = f'{protocol}://{username}:{password}@{host}:{port}/{dbname}'
        self.engine = create_engine(self.url)
        self.session_local = sessionmaker(
            autocommit=False, autoflush=False, bind=self.engine
        )

    @contextmanager
    def get_session(self) -> Generator[Session, None, None]:
        """
        Provides a session generator that yields a database session and ensures it is
        closed after use.

        Yields:
            Session: SQLAlchemy database session.
        """
        db = self.session_local()
        try:
            yield db
        finally:
            db.close()

    def compile_sql(self, query: Query) -> str:
        """
        Compiles a SQLAlchemy Query object to its raw SQL string representation.

        Args:
            query: The SQLAlchemy Query object to compile.
        """
        return str(
            query.statement.compile(
                dialect=postgresql.dialect(), compile_kwargs={'literal_binds': True}
            )
        )


class DatabaseManager:
    def __init__(self):
        # Dictionary to hold Database instances keyed by DatabaseName
        self.databases: Dict[DatabaseName, Database] = {}

    def get(self, database_name: DatabaseName) -> Database:
        """
        Retrieves the Database instance for the specified database name.

        Args:
            database_name: The name of the database to retrieve.

        Returns:
            Database: The corresponding Database instance.
        """
        return self.databases[database_name]
