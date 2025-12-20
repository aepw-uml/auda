from collections import defaultdict
from logging import Logger
from typing import Dict, List, Tuple, cast

from auda.core import Database, DatabaseName, auda
from auda.model import DataColumnMetadata, TableMetadata
from sqlalchemy import (
    BigInteger,
    Column,
    Engine,
    Float,
    Integer,
    MetaData,
    String,
    Table,
    Text,
    inspect,
)


class DataService:
    def __init__(self):
        """Initializes a DataService instance.

        Attributes:
            logger: Logger instance for logging messages.
            auda_db: Database instance for accessing the AUDA database.
        """

        self.logger: Logger = auda.get_logger(__class__.__name__)
        self.auda_db: Database = auda.database_manager.get(DatabaseName.AUDA)

    def get_data_column_metadata(self) -> List[DataColumnMetadata]:
        """Retrieves data column metadata from the database."""

        with self.auda_db.get_session() as session:
            return session.query(DataColumnMetadata).all()

    def get_data_column_metadata_map(self) -> Dict[str, DataColumnMetadata]:
        """Returns a mapping of original data column names to their
        corresponding metadata objects.
        """

        return {
            str(metadata.original_column_name): metadata
            for metadata in self.get_data_column_metadata()
        }

    def get_data_table_map(self) -> Dict[str, Table]:
        """Returns a mapping of data table names to their corresponding
        SQLAlchemy Table objects.
        """

        with self.auda_db.get_session() as session:
            table_list = session.query(TableMetadata).all()

        return {
            table.name: Table(
                table.name, MetaData(), autoload_with=session.bind
            )
            for table in table_list
            if table.type == 'data'
        }

    def _map_data_type(self, data_type: str):
        """Maps the string in data_column_metadata.data_type to a SQLAlchemy
        type.

        Args:
            data_type: The data type as a string.

        Returns:
            A SQLAlchemy type.
        """

        key = (data_type or '').strip().lower()

        if key in {'int', 'integer'}:
            return BigInteger
        if key in {'float', 'double', 'real', 'decimal', 'numeric', 'number'}:
            return Float
        if key in {'string', 'varchar', 'char'}:
            return String(255)
        if key in {'text'}:
            return Text

        return Text

    def gather_columns_by_table(
        self, rows: List[DataColumnMetadata]
    ) -> Dict[str, List[Tuple[str, object, str]]]:
        """Gathers columns by their respective table names.

        Args:
            rows: A list of DataColumnMetadata objects.

        Returns:
            A dictionary mapping table names to lists of tuples, each
            containing column name, SQLAlchemy type, and column comment.
        """

        sorted_rows = sorted(rows, key=lambda r: getattr(r, 'id', 0))
        grouped: Dict[str, List[Tuple[str, object, str]]] = defaultdict(list)

        for row in sorted_rows:
            column_type = self._map_data_type(
                str(getattr(row, 'data_type', None))
            )
            column_name = getattr(row, 'column_name')
            column_comment = getattr(row, 'description', '') or ''
            grouped[getattr(row, 'table_name')].append(
                (column_name, column_type, column_comment)
            )

        return grouped

    def create_data_tables(self) -> None:
        """Creates data tables in the database based on the data column
        metadata.
        """

        columns_by_table = self.gather_columns_by_table(
            self.get_data_column_metadata()
        )

        with self.auda_db.get_session() as session:
            engine = cast(Engine, session.get_bind())
            inspector = inspect(engine)
            metadata = MetaData()

            for table_name, columns in columns_by_table.items():
                if inspector.has_table(table_name):
                    self.logger.info(
                        f"Table '{table_name}' already exists; skipping."
                    )
                    continue

                sa_columns: List[Column] = [
                    Column('location', String(255), nullable=False),
                    Column('year', Integer, nullable=False),
                ]
                for column_name, column_type, column_comment in columns:
                    # You can adjust nullable=True/False based on your needs
                    sa_columns.append(
                        Column(
                            column_name,
                            column_type,  # type: ignore
                            nullable=True,
                            comment=column_comment or None,
                        )
                    )

                tbl = Table(table_name, metadata, *sa_columns, comment=None)

                self.logger.info(
                    f"Creating table '{table_name}' with {len(sa_columns)} "
                    'columns.'
                )
                tbl.create(bind=engine, checkfirst=True)

    def drop_data_tables(self) -> None:
        """Drops data tables in the database based on the data column
        metadata.
        """

        columns_by_table = self.gather_columns_by_table(
            self.get_data_column_metadata()
        )

        with self.auda_db.get_session() as session:
            engine = cast(Engine, session.get_bind())
            inspector = inspect(engine)
            metadata = MetaData()

            for table_name in columns_by_table.keys():
                if not inspector.has_table(table_name):
                    self.logger.info(
                        f"Table '{table_name}' does not exist; skipping."
                    )
                    continue

                tbl = Table(table_name, metadata, autoload_with=engine)

                self.logger.info(f"Dropping table '{table_name}'.")
                tbl.drop(bind=engine, checkfirst=True)

    def get_data_table_columns(self) -> Dict[str, List[str]]:
        """Retrieves the columns of each data table in the database."""

        data_table_map = self.get_data_table_map()

        result = defaultdict(list)
        with self.auda_db.get_session() as session:
            inspector = inspect(session.get_bind())
            for table_name in data_table_map.keys():
                if inspector.has_table(table_name):
                    result[table_name] = inspector.get_columns(table_name)

        return result

    def get_countries(self) -> List[str]:
        """Retrieves a list of distinct countries from the data tables.

        Returns:
            A list of country names.
        """

        countries = set()
        data_table_map = self.get_data_table_map()

        with self.auda_db.get_session() as session:
            table = data_table_map['d_demography']
            query = session.query(table.c.location).distinct()
            for row in query.all():
                countries.add(row.location)

        return sorted(countries)
