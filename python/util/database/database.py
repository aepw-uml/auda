from contextlib import contextmanager
from typing import Generator

from sqlalchemy import Engine, create_engine
from sqlalchemy.dialects import postgresql
from sqlalchemy.orm import Query, Session, sessionmaker


class Database:
    """Represents a database connection and session manager using SQLAlchemy."""

    def __init__(self, url: str):
        """Initializes the database connection URL, engine, and session factory.

        Args:
            url: The database connection URL.

        Attributes:
            url: The database connection URL.
            engine: The SQLAlchemy engine instance.
            session_local: The session factory for creating database sessions.
        """

        self.url: str = url
        self.engine: Engine = create_engine(url)
        self.session_local: sessionmaker = sessionmaker(
            autocommit=False, autoflush=False, bind=self.engine
        )

    @contextmanager
    def get_session(self) -> Generator[Session, None, None]:
        """Provides a session generator that yields a database session.

        This method ensures that the database session is closed after use.

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
                dialect=postgresql.dialect(),
                compile_kwargs={'literal_binds': True},
            )
        )
