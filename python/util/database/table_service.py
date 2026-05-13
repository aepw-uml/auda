from typing import Any, Callable, Dict, List, Optional, Tuple

from pydantic import BaseModel
from sqlalchemy import (
    Column,
    Function,
    Join,
    MetaData,
    Select,
    Table,
    and_,
    func,
    select,
)
from sqlalchemy.dialects import postgresql
from util.database.database import Database

from .model import DataColumnMetadata, TableMetadata
from .pagination import Pagination

# Define aggregate functions mapping
AGGREGATE_FUNCTIONS = {
    'sum': func.sum,
    'avg': func.avg,
    'count': func.count,
}


class AggregateEntry(BaseModel):
    """Represents a group by entry in a query.

    Attributes:
        group_by_column_names: List of column names to group by; can only
        contain 'year' and 'location'.
        aggregate_fn: The aggregate function to apply.
    """

    group_by_column_names: List[str]
    aggregate_fn: str


class SortEntry(BaseModel):
    """Represents a sort entry in a query.

    Attributes:
        column_name: The name of the column to sort by.
        direction: The direction of sorting; either 'asc' for ascending or
            'desc' for descending.
    """

    column_name: str
    direction: str


class TableQueryParams(BaseModel):
    """Represents the parameters for a table query.

    Attributes:
        column_names: List of column names to retrieve; if None, all columns are
            retrieved.
        sort_entries: List of SortEntry objects specifying how to sort the
            results.
        notnull_column_names: List of column names to filter out null values;
            if None, no filtering is applied.
        pagination: Pagination object specifying the page and page size for the
            results.
        aggregate_entry: AggregateEntry object specifying how to group and
            aggregate the results; if None, no aggregation is applied.
    """

    column_names: Optional[List[str]] = None
    sort_entries: Optional[List[SortEntry]] = None
    notnull_column_names: Optional[List[str]] = None
    pagination: Optional[Pagination] = None
    aggregate_entry: Optional[AggregateEntry] = None


class TableResult(BaseModel):
    """
    Represents the result of a table query.

    Attributes:
        column_names: List of column names in the result.
        data: List of records, where each record is a list of values
            corresponding to the columns.
        pagination: Pagination object specifying the page and page size for the
            results.
        sql: The SQL query string used to retrieve the results.
    """

    column_names: List[str]
    data: List[List[Any]]
    pagination: Pagination
    sql: str


class TableService:
    def __init__(self, database_url: str):
        """Initializes the TableService with the specified database.

        Args:
            database_url: The URL of the database to connect to.

        Attributes:
            db: The database connection object.
        """

        self.db = Database(database_url)

    def get_all_table_metadata(self) -> List[TableMetadata]:
        """Retrieves all retrievable tables from the database.

        All retrievable tables are listed in the TableMetadata table.
        """

        with self.db.get_session() as session:
            return session.query(TableMetadata).all()

    def get_metadata(self) -> MetaData:
        """Retrieves the metadata for all tables in the database.

        Returns:
            The SQLAlchemy MetaData object containing metadata for all tables.
        """

        metadata = MetaData()
        metadata.reflect(bind=self.db.engine)

        return metadata

    def get_table_metadata(self, table_name: str) -> Optional[TableMetadata]:
        """Retrieves metadata for a specific data table.

        Args:
            table_name: The name of the table to retrieve metadata for.

        Returns:
            The metadata for the specified table, or None if not found.
        """

        with self.db.get_session() as session:
            return (
                session.query(TableMetadata)
                .filter(TableMetadata.name == table_name)
                .first()
            )

    def get_ui_column_map(self, table_name: str) -> Dict[str, str]:
        """Retrieves a mapping of column names to their UI column names for a
        specific data table.

        Args:
            table_name: The name of the table to retrieve the UI column map for.

        Returns:
            A dictionary mapping column names to their UI column names.
        """

        table_metadata: Optional[TableMetadata] = self.get_table_metadata(
            table_name
        )
        if table_metadata is None:
            raise ValueError(f"Table metadata for '{table_name}' not found.")

        if table_metadata.type == 'data':
            column_metadata_list: List[DataColumnMetadata] = (
                self.get_data_table_column_metadata(table_name)
            )
            ui_column_map = {
                cm.column_name: cm.ui_column_name for cm in column_metadata_list
            }
        else:
            # The table is not a data table, then we return a mapping where the
            # column name is the same as the UI column name
            metadata = MetaData()
            table = Table(table_name, metadata, autoload_with=self.db.engine)
            column_names = [column.name for column in table.columns]
            ui_column_map = {
                column_name: column_name for column_name in column_names
            }

        return ui_column_map

    def get_data_table_column_metadata(
        self,
        table_name: str,
    ) -> List[DataColumnMetadata]:
        """Retrieves metadata for the columns of a specific data table.

        Args:
            table_name: The name of the data table to retrieve column metadata
                for.

        Returns:
            A list of DataColumnMetadata objects containing metadata for the
            columns of the specified data table.
        """

        with self.db.get_session() as session:
            return (
                session.query(DataColumnMetadata)
                .filter(DataColumnMetadata.table_name == table_name)
                .all()
            )

    def prepare_tables(
        self, table_names: List[str]
    ) -> Tuple[List[Table], Dict[str, TableMetadata]]:
        """Prepares the tables for querying by loading their metadata and
        SQLAlchemy Table objects.

        Args:
            table_names: A list of table names to prepare.

        Returns:
            A tuple containing:
                - A list of SQLAlchemy Table objects for the specified table
                    names.
                - A dictionary mapping table names to their metadata.
        """

        # Edge case: if no table names are provided, raise an exception
        if not table_names:
            raise ValueError('At least one table name must be specified.')

        # Retrieve metadata for all table names
        table_metadata_map: Dict[str, TableMetadata] = {
            table_name: self.get_table_metadata(table_name)
            for table_name in table_names
        }

        # Validate the table metadata map
        self.validate_table_metadata_map(table_metadata_map)

        # Load SQLAlchemy tables and generate a list of SQLAlchemy Table objects
        metadata = MetaData()
        tables: List[Table] = [
            self.load_table(table_name, metadata) for table_name in table_names
        ]

        return tables, table_metadata_map

    def get_column_map(self, tables: List[Table]) -> Dict[str, Column]:
        """Generates a mapping of column names to SQLAlchemy Column objects for
        the provided list of tables.

        Args:
            tables: A list of SQLAlchemy Table objects.

        Returns:
            A dictionary mapping column names (with table aliasing) to their
            corresponding SQLAlchemy Column objects.
        """

        return {
            f'{tbl.name}.{col.name}': col
            for tbl in tables
            for col in tbl.columns
        }

    def get_record_count(
        self,
        tables: List[Table],
        table_query_params: TableQueryParams,
    ) -> int:
        """Retrieves the total number of records in the specified tables that
        match the given query parameters.
        """
        table: Table | Join = self.create_joined_table(tables)
        column_map: Dict[str, Column] = self.get_column_map(tables)
        query: Select = select(func.count()).select_from(table)
        query = self.update_query_with_notnull_column_names(
            query, column_map, table_query_params.notnull_column_names
        )

        with self.db.get_session() as session:
            return session.execute(query).scalar_one()

    def get_table_result(
        self,
        tables: List[Table],
        table_metadata_map: Dict[str, TableMetadata],
        table_query_params: TableQueryParams,
        query_updater: Optional[Callable[[Select], Select]] = None,
    ) -> TableResult:
        """Retrieves the table result based on the provided tables and query
        parameters.

        Args:
            tables: A list of SQLAlchemy Table objects to query.
            table_metadata_map: A dictionary mapping table names to their
                metadata.
            table_query_params: The parameters for querying the tables.
            query_updater: An optional callable that takes a SQLAlchemy Select
                query and returns an updated Select query.

        Returns:
            The result of the table query as a TableResult object.
        """

        table: Table | Join = self.create_joined_table(tables)
        column_map: Dict[str, Column] = self.get_column_map(tables)
        all_column_names: List[str] = list(column_map.keys())

        # Table query parameters
        column_names: List[str] = table_query_params.column_names or []
        sort_entries: Optional[List[SortEntry]] = (
            table_query_params.sort_entries
        )
        notnull_column_names: Optional[List[str]] = (
            table_query_params.notnull_column_names
        )
        pagination: Pagination = table_query_params.pagination or Pagination(
            page=1, page_size=-1
        )
        aggregate_entry: Optional[AggregateEntry] = (
            table_query_params.aggregate_entry
        )

        # If the `column_names` is not an empty list, then use its shallow copy
        # as the `selected_column_names` list; otherwise, use the shallow copy
        # of the `all_column_names` list
        selected_column_names: List[str] = (
            column_names[:] if column_names else all_column_names[:]
        )

        # Create a list of SQLAlchemy columns based on the selected column
        # names
        columns: List[Column] = []
        for column_name in selected_column_names:
            if column_name in column_map:
                columns.append(column_map[column_name])

        if aggregate_entry:
            if not column_names:
                raise ValueError(
                    'Column names must be specified when using aggregation.',
                )

            group_by_columns, aggregate_function, selected_column_names = (
                self.get_aggregate_columns_and_functions(
                    column_map,
                    aggregate_entry,
                    column_names[0],
                    selected_column_names,
                )
            )
            query: Select = (
                select(*group_by_columns, aggregate_function)
                .select_from(table)
                .group_by(*group_by_columns)
            )
        else:
            query: Select = select(*columns).select_from(table)

        # Update the select query
        query = self.update_query_with_sort_entries(
            query, column_map, sort_entries
        )
        query = self.update_query_with_notnull_column_names(
            query, column_map, notnull_column_names
        )
        query = self.update_query_with_pagination(query, pagination)

        # Update the select query with the query updater if provided
        if query_updater:
            query = query_updater(query)

        # Execute the query and fetch all results
        with self.db.get_session() as session:
            result = session.execute(query)
        records = [list(row) for row in result.fetchall()]

        # Postprocess the table columns and records
        if (
            len(table_metadata_map) > 1
            and next(iter(table_metadata_map.values())).type == 'data'
        ):
            selected_column_names, records = (
                self.postprocess_table_columns_records(
                    selected_column_names, records
                )
            )

        sql = str(query.compile(compile_kwargs={'literal_binds': True}))

        return TableResult(
            column_names=selected_column_names,
            data=records,
            pagination=pagination,
            sql=sql,
        )

    def prepare_get_table_result(
        self,
        table_names: List[str],
        table_query_params: TableQueryParams,
        query_updater: Optional[Callable[[Select], Select]] = None,
    ) -> TableResult:
        """
        Prepares the tables and retrieves the table result based on the provided
        table names and query parameters.

        Args:
            table_names: A list of table names to prepare and query.
            table_query_params: The parameters for querying the tables.

        Returns:
            The result of the table query as a TableResult object.
        """

        tables, table_metadata_map = self.prepare_tables(table_names)
        return self.get_table_result(
            tables, table_metadata_map, table_query_params, query_updater
        )

    def validate_table_metadata_map(
        self, table_metadata_map: Dict[str, TableMetadata]
    ) -> bool:
        """
        Validates the table metadata map to ensure that it either:

        - Contains only one table (no joins)
        - Contains only data tables (for joins)

        Args:
            table_metadata_map: A dictionary mapping table names to their
                metadata.

        Returns:
            True if the table metadata map is valid for the intended operation.
        """

        if len(table_metadata_map) == 1:
            table_metadata_list = list(table_metadata_map.values())
            return table_metadata_list[0].type == 'data'

        for table_name, metadata in table_metadata_map.items():
            if metadata.type != 'data':
                raise ValueError(
                    f"Table '{table_name}' is not a data table and cannot be "
                    'joined.',
                )

        return True

    def load_table(self, table_name: str, metadata: MetaData) -> Table:
        """Loads a table from the database using SQLAlchemy's MetaData.

        Args:
            table_name: The name of the table to load.
            metadata: The SQLAlchemy MetaData object.

        Returns:
            The SQLAlchemy Table object for the specified table.
        """

        return Table(table_name, metadata, autoload_with=self.db.engine)

    def build_column_name_list(self, tables: List[Table]) -> List[str]:
        """Builds a list of column names with table name aliasing to avoid
        ambiguity in SQL queries.

        Args:
            tables: A list of SQLAlchemy Table objects.

        Returns:
            A list of column names with table name aliasing.

        Example:
            If `table1` has `col1` and `col2`, and `table2` has `col3` and
            `col4`, the `all_column_names` list will be:

            ['table1.col1', 'table1.col2', 'table2.col3', 'table2.col4']
        """

        column_names: List[str] = []
        for table in tables:
            column_names.extend(
                [f'{table.name}.{col.name}' for col in table.columns]
            )

        return column_names

    def create_joined_table(self, tables: List[Table]) -> Table | Join:
        """Creates a joined table if the number of tables is greater than 1;
        otherwise, returns the first table in the list.

        It is assumed that if multiple tables are provided, they are data
        tables, and their primary keys are "year" and "location".

        The Equivalent SQL would be as follows:

             SELECT ? FROM t0
             JOIN t1 ON t0.year = t1.year AND t0.location = t1.location
             JOIN t2 ON t0.year = t2.year AND t0.location = t2.location
             ...

        Args:
            tables: A list of SQLAlchemy tables.

        Returns:
            A joined table.
        """

        if len(tables) == 1:
            return tables[0]

        main_table = tables[0]
        joined_table = main_table
        for other_table in tables[1:]:
            joined_table = joined_table.join(
                other_table,
                and_(
                    main_table.c.year == other_table.c.year,
                    main_table.c.location == other_table.c.location,
                ),
            )

        return joined_table

    def get_column(
        self,
        column_name: str,
        column_map: Dict[str, Column],
        context: Optional[str],
    ) -> Column:
        """Validates the column name and returns the corresponding SQLAlchemy
        Column object.

        Args:
            column_name: The name of the column to validate.
            column_map: A mapping of column names to SQLAlchemy Column objects.

        Returns:
            The validated column name and the corresponding SQLAlchemy Column
            object.

        Raises:
            ValueError: If the column name is not found in the column map.
        """

        context = f'in {context} ' if context else ''
        if column_name not in column_map:
            raise ValueError(
                f"Column '{column_name}' {context}not found in column map.",
            )

        return column_map[column_name]

    def update_query_with_sort_entries(
        self,
        query: Select,
        column_map: Dict[str, Column],
        sort_entries: Optional[List[SortEntry]],
    ) -> Select:
        """Updates the SQLAlchemy query with sorting based on the provided sort
        entries.

        Args:
            query: The SQLAlchemy Select query to update.
            column_map: A mapping of column names to SQLAlchemy Column objects.
            sort_entries: A list of SortEntry objects specifying how to sort the
                results.
        """

        if not sort_entries:
            return query

        for entry in sort_entries:
            column = self.get_column(entry.column_name, column_map, 'sort')
            if entry.direction == 'desc':
                query = query.order_by(column.desc())
            else:
                query = query.order_by(column.asc())

        return query

    def update_query_with_notnull_column_names(
        self,
        query: Select,
        column_map: Dict[str, Column],
        notnull_column_names: Optional[List[str]],
    ) -> Select:
        """Updates the SQLAlchemy query to filter out null values for the
        specified column names.

        Args:
            query: The SQLAlchemy Select query to update.
            column_map: A mapping of column names to SQLAlchemy Column objects.
            notnull_column_names: A list of column names to filter out null
                values.
        """

        if not notnull_column_names:
            return query

        real_notnull_column_names: List[str] = []
        for column_name in notnull_column_names:
            if column_name == 'year' or column_name == 'location':
                continue

            if '.' in column_name:
                _, _column_name = column_name.split('.')
                if _column_name == 'year' or _column_name == 'location':
                    continue

            real_notnull_column_names.append(column_name)

        for column_name in real_notnull_column_names:
            column = self.get_column(column_name, column_map, 'notnull filter')
            query = query.filter(
                and_(column.is_not(None), column != float('nan'))
            )

        return query

    def update_query_with_pagination(
        self, query: Select, pagination: Optional[Pagination]
    ) -> Select:
        """Updates the SQLAlchemy query with pagination based on the provided
        pagination parameters.

        Args:
            query: The SQLAlchemy Select query to update.
            pagination: A Pagination object specifying the page and page size
                for the results.
        """

        if pagination is None:
            return query

        page: int = pagination.page
        page_size: int = pagination.page_size
        offset: int = (page - 1) * page_size

        if offset > 0:
            query = query.offset(offset)

        if page_size > 0:
            query = query.limit(page_size)

        return query

    def get_aggregate_columns_and_functions(
        self,
        column_map: Dict[str, Column],
        aggregate_entry: AggregateEntry,
        column_name: str,
        selected_column_names: List[str],
    ):
        """Retrieves the group by columns and aggregate function based on the
        provided aggregate entry.

        Args:
            column_map: A mapping of column names to SQLAlchemy Column objects.
            aggregate_entry: An AggregateEntry object specifying how to group
                and aggregate the results.
            column_name: The name of the column to aggregate.
            selected_column_names: A list of selected column names.
        """

        group_by_column_names: List[str] = aggregate_entry.group_by_column_names
        selected_column_names = selected_column_names[:]
        aggregate_column_name = column_name

        # Group by columns must be specified
        if not group_by_column_names:
            raise ValueError('Columns must be specified when using group by.')

        group_by_columns: List[Column] = []
        for group_by_column_name in group_by_column_names:
            # Ensure that the group by columns are valid (either year or
            # location)
            if (
                group_by_column_name != 'year'
                and group_by_column_name != 'location'
            ):
                raise ValueError(
                    'Group by column names must be year or location.'
                )

            # Get column from any table
            for table_column_name, column in column_map.items():
                _, column_name = table_column_name.split('.')
                if column_name == group_by_column_name:
                    group_by_columns.append(column)
                    selected_column_names.insert(0, table_column_name)
                    break

        aggregate_column = self.get_column(
            aggregate_column_name, column_map, 'aggregate'
        )
        aggregate_function: Function = self.validate_aggregate_function(
            aggregate_entry.aggregate_fn, aggregate_column
        )

        return group_by_columns, aggregate_function, selected_column_names

    def validate_aggregate_function(
        self, function_name: str, aggregate_column: Column
    ) -> Function:
        """Validates the aggregation function name and returns the corresponding
        SQLAlchemy function.

        Args:
            function_name: The name of the aggregation function to validate.
            aggregate_column: The SQLAlchemy Column object to apply the
                aggregation function on.

        Returns:
            The validated aggregation function as a SQLAlchemy Function object.
        """
        if function_name not in AGGREGATE_FUNCTIONS:
            raise ValueError(
                f'Unsupported aggregation function {function_name}.'
            )

        return AGGREGATE_FUNCTIONS[function_name](aggregate_column).label(
            f'{function_name}_{aggregate_column.name}'
        )

    def postprocess_table_columns_records(
        self, column_names: List[str], records: List[List[str]]
    ) -> Tuple[List[str], List[List[str]]]:
        """Postprocesses the table columns and records to ensure that there is
        only one year and one location column, and to map the column names to
        their UI column names.

        Args:
            column_names: A list of column names in the format 'table.column'.
            records: A list of records, where each record is a list of values
                corresponding to the columns.

        Returns:
            A tuple containing:
                - A list of new column names after postprocessing.
                - A list of records with the selected columns.
        """

        # Mapping from table name to a column name mappings
        # Each column name mapping maps a column name to its UI column name
        has_year_appeared = False
        has_location_appeared = False
        selected_column_indexes: List[int] = []
        new_column_names: List[str] = []
        for index, column_name in enumerate(column_names):
            _, _column_name = column_name.split('.')

            # Make sure that there are only one year/location columns
            if _column_name == 'year':
                if not has_year_appeared:
                    has_year_appeared = True
                    selected_column_indexes.append(index)
                    new_column_names.append(column_name)

                continue

            if _column_name == 'location':
                if not has_location_appeared:
                    has_location_appeared = True
                    selected_column_indexes.append(index)
                    new_column_names.append(column_name)

                continue

            new_column_names.append(column_name)
            selected_column_indexes.append(index)

        new_records = []
        for record in records:
            new_record = [record[i] for i in selected_column_indexes]
            new_records.append(new_record)

        return new_column_names, new_records

    def get_sql(self, query: Select) -> str:
        """Compiles the SQLAlchemy query into a SQL string with literal binds.

        Args:
            query: The SQLAlchemy Select query to compile.

        Returns:
            A string representation of the SQL query with literal binds.
        """

        return str(
            query.compile(
                dialect=postgresql.dialect(),
                compile_kwargs={'literal_binds': True},
            )
        )
