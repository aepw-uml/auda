import math
from decimal import Decimal, InvalidOperation
from logging import Logger
from typing import Any

from sqlalchemy import (
    BigInteger,
    Float,
    Integer,
    MetaData,
    Select,
    SmallInteger,
    Table,
    and_,
    func,
    insert,
    select,
    update,
)
from util.database import Database
from util.database.model import (
    DataColumnMetadata,
    PrismDataExtraction,
    PrismDataPoint,
    PrismLocation,
    TableMetadata,
)
from util.env import env
from util.logging import get_logger

from .datapoint import Datapoint
from .migration_options import MigrationOptions


class MigrationService:
    def __init__(self, options: MigrationOptions) -> None:
        """Initializes the MigrationService with the given migration options.

        Args:
            options: The migration options to use for the migration process.
        """

        self.options: MigrationOptions = options
        self.logger: Logger = get_logger('MigrationService')
        self.auda_db: Database = Database(env.dbUrl)
        self.prism_db: Database = Database(env.prismDbUrl)

    def migrate(self) -> None:
        batch_size: int = self.options.batch_size
        is_validated: bool = self.options.is_validated
        allow_duplicates: bool = self.options.allow_duplicates
        minimal_completeness_score: int = (
            self.options.minimal_completeness_score
        )

        self.logger.info(f'Migration Batch size is set to {batch_size}.')
        self.logger.info(
            f'Migration only includes validated datapoints: {is_validated}.'
        )
        self.logger.info(
            f'Migration allows duplicated datapoints: {allow_duplicates}.'
        )
        self.logger.info(
            f'Migration minimal completeness score is set to '
            f'{minimal_completeness_score}.'
        )

        self.logger.info('Preparing migration...')

        num_datapoints = self.get_num_datapoints()
        self.logger.info(
            f'Found {num_datapoints} migratable datapoints in the PRISM '
            'database.'
        )

        data_columns_by_ids = self.get_data_column_map()
        self.logger.info(f'Found {len(data_columns_by_ids)} data column types.')

        locations_by_ids: dict[str, PrismLocation] = self.get_location_map()
        self.logger.info(f'Found {len(locations_by_ids)} locations.')

        column_metadata_by_names: dict[str, DataColumnMetadata] = (
            self.get_data_column_metadata_map()
        )
        self.logger.info(
            f'Found {len(column_metadata_by_names)} data column metadata.'
        )

        data_tables_by_names: dict[str, Table] = self.get_data_table_map()
        data_table_names: list[str] = list(data_tables_by_names.keys())
        self.logger.info(f'Prepared {len(data_table_names)} data tables.')

        # Construct base query by selecting specific columns to avoid loading
        # unnecessary data.
        base_query = select(
            PrismDataExtraction.id,
            PrismDataExtraction.data_point_id,
            PrismDataExtraction.location_id,
            PrismDataExtraction.year,
            PrismDataExtraction.value,
            PrismDataExtraction.unit,
        ).select_from(PrismDataExtraction)
        base_query = self.update_query(base_query)

        # Sort the query by ID to ensure consistent ordering.
        base_query = base_query.order_by(PrismDataExtraction.id.asc())

        # Calculate the number of batches.
        batch_size: int = self.options.batch_size
        num_batches = math.ceil(num_datapoints / batch_size)
        self.logger.info(f'Total batches to process: {num_batches}.')

        batch_index: int = 0
        migrated_count: int = 0
        skipped_count: int = 0
        inserted_count: int = 0

        while migrated_count < num_datapoints:
            self.logger.info(
                f'[{batch_index + 1}/{num_batches}] Migrating batch...'
            )

            datapoints: list[Datapoint] = self.fetch_datapoints(
                base_query, batch_size, batch_index
            )
            self.logger.debug(f'Fetched {len(datapoints)} datapoints.')

            for datapoint in datapoints:
                inserted: bool = self.migrate_datapoint(
                    datapoint,
                    data_columns_by_ids,
                    locations_by_ids,
                    column_metadata_by_names,
                    data_tables_by_names,
                )

                if inserted:
                    inserted_count += 1
                else:
                    skipped_count += 1

            batch_index += 1
            migrated_count += batch_size

        self.logger.info('Migration completed.')
        self.logger.info(f'Total datapoints migrated: {inserted_count}.')
        self.logger.info(f'Total datapoints skipped: {skipped_count}.')

    def update_query(self, query: Select) -> Select:
        """Updates the given SQLAlchemy query based on the migration options.

        The ``validation_status`` can be either VALIDATED or INVALIDATED. The
        ``data_completeness_score`` is an integer between 0 and 100.
        """

        if self.options.is_validated:
            query = query.where(
                PrismDataExtraction.validation_status == 'VALIDATED'
            )
        if not self.options.allow_duplicates:
            query = query.where(PrismDataExtraction.is_duplicated.is_(False))
        if self.options.minimal_completeness_score > 0:
            query = query.where(
                PrismDataExtraction.data_completeness_score
                >= self.options.minimal_completeness_score
            )
        if self.options.minimal_data_source_score > 0:
            query = query.where(
                PrismDataExtraction.data_source_score
                >= self.options.minimal_data_source_score
            )

        # Filter out records where `unit = 'NA'`
        query = query.where(PrismDataExtraction.unit != 'NA')

        return query

    def get_num_datapoints(self) -> int:
        """Retrieves the number of migratable datapoints in the PRISM database
        based on the migration options.
        """

        query = select(func.count()).select_from(PrismDataExtraction)
        query = self.update_query(query)

        with self.prism_db.get_session() as prism_session:
            return prism_session.execute(query).scalar_one()

    def get_data_column_map(self) -> dict[str, PrismDataPoint]:
        """Retrieves a mapping of the IDs in the PrismDataPoint (data column
        metadata) table to the corresponding PrismDataPoint objects from the
        PRISM database.
        """

        query = select(PrismDataPoint)

        with self.prism_db.get_session() as prism_session:
            result = prism_session.execute(query).scalars().all()
            return {str(datapoint.id): datapoint for datapoint in result}

    def get_location_map(self) -> dict[str, PrismLocation]:
        """Retrieves a mapping of the IDs in the PrismLocation table to the
        corresponding PrismLocation objects from the PRISM database.
        """

        query = select(PrismLocation)

        with self.prism_db.get_session() as prism_session:
            result = prism_session.execute(query).scalars().all()
            return {str(location.id): location for location in result}

    def get_data_column_metadata_map(self) -> dict[str, DataColumnMetadata]:
        """Retrieves a mapping of the IDs in the DataColumnMetadata table to the
        corresponding DataColumnMetadata objects from the PRISM database.
        """

        query = select(DataColumnMetadata)

        with self.auda_db.get_session() as auda_session:
            result = auda_session.execute(query).scalars().all()
            return {
                metadata.original_column_name: metadata for metadata in result
            }

    def get_data_table_map(self) -> dict[str, Table]:
        """Retrieves a mapping of the names of the data tables to the
        corresponding SQLAlchemy Table objects from the AUDA database.
        """

        with self.auda_db.get_session() as auda_session:
            tables = auda_session.query(TableMetadata).all()
            return {
                table.name: Table(
                    table.name, MetaData(), autoload_with=auda_session.bind
                )
                for table in tables
                if table.type == 'data'
            }

    def fetch_datapoints(
        self, base_query: Select, batch_size: int, batch_index: int
    ) -> list[Datapoint]:
        """Fetches a batch of datapoints from the PRISM database based on the
        given base query, batch size, and batch index.

        The base query should already include the necessary filters based on the
        migration options. This method will apply the appropriate limit and
        offset to fetch the correct batch of datapoints for migration.
        """

        batch_query = base_query.limit(batch_size).offset(
            batch_size * batch_index
        )

        with self.prism_db.get_session() as prism_session:
            batch_data = prism_session.execute(batch_query).fetchall()
            datapoints = [
                Datapoint(
                    id=str(row.id),
                    datapoint_id=str(row.data_point_id),
                    location_id=str(row.location_id),
                    year=row.year,
                    value=str(row.value),
                    unit=str(row.unit).strip(),
                )
                for row in batch_data
            ]

            return datapoints

    def migrate_datapoint(
        self,
        datapoint: Datapoint,
        data_columns_by_ids: dict[str, PrismDataPoint],
        locations_by_ids: dict[str, PrismLocation],
        column_metadata_by_names: dict[str, DataColumnMetadata],
        data_tables_by_names: dict[str, Table],
    ) -> bool:
        """Migrates a single datapoint to the AUDA database. Returns True if
        the datapoint was successfully migrated, or False if it was skipped.

        Returns:
            True if the datapoint was successfully migrated, False if it was
            skipped.
        """

        data_point_id = datapoint.datapoint_id
        data_column = data_columns_by_ids.get(data_point_id)
        if data_column is None:
            return False

        original_column_name: str = data_column.name
        column_metadata = column_metadata_by_names.get(original_column_name)
        if column_metadata is None:
            self.logger.debug(
                f"Skipping datapoint with data column '{original_column_name}' "
                'because no matching column metadata was found.'
            )
            return False

        table_name: str = column_metadata.table_name
        data_table = data_tables_by_names.get(table_name)
        if data_table is None:
            self.logger.debug(
                f"Skipping datapoint with data column '{original_column_name}' "
                f"because target table '{table_name}' was not found."
            )
            return False

        # Get the location name.
        prism_location = locations_by_ids.get(datapoint.location_id)
        if prism_location is None:
            self.logger.debug(
                f"Skipping datapoint with location ID '{datapoint.location_id}"
                "' because no matching location was found."
            )
            return False

        # The primary key of all data tables.
        location: str = prism_location.location_name
        year: int = datapoint.year

        # Insert or update the datapoint.
        return self.insert_or_update_datapoint(
            data_table,
            column_metadata.column_name,
            location,
            year,
            datapoint.value,
        )

    def insert_or_update_datapoint(
        self,
        data_table: Table,
        column_name: str,
        location: str,
        year: int,
        value: Any,
    ) -> bool:
        """Checks if the given data table contains a record whose primary key is
        (location, year). If it does, update the value of the specified column.
        If not, insert a new record, settings all other columns to NULL.

        Returns:
            True if the datapoint was successfully inserted or updated, False if
            there was an error (e.g. the target column doesn't exist, or the
            value couldn't be coerced to the right type).
        """

        # Assume that the target column exists on the data table.
        target_col = data_table.c[column_name]

        try:
            location_col = data_table.c['location']
            year_col = data_table.c['year']
        except KeyError as ex:
            raise ValueError(
                "Expected PK columns 'location' and 'year' not found on table"
                f"'{data_table.name}'"
            ) from ex

        # Type-aware coercion (handles '1.79E+11', commas, etc.)
        value = self._coerce_for_column(target_col.type, value)
        if value is None:
            self.logger.debug(
                f"Skipping datapoint for location '{location}' and year "
                f"'{year}' because the value is None after coercion."
            )
            return False

        where_primary_key = and_(location_col == location, year_col == year)
        with self.auda_db.get_session() as ada_db_session:
            with ada_db_session.begin():
                # First, try to update an existing record.
                upd_stmt = (
                    update(data_table)
                    .where(where_primary_key)
                    .values({target_col.key: value})
                )
                result = ada_db_session.execute(upd_stmt)

                # If no record was updated, insert a new one.
                if result.rowcount == 0:  # type: ignore
                    insert_values = {
                        location_col.key: location,
                        year_col.key: year,
                        target_col.key: value,
                    }
                    insert_stmt = insert(data_table).values(insert_values)
                    ada_db_session.execute(insert_stmt)

        return True

    def _coerce_for_column(self, column_type: Any, raw: Any) -> Any:
        """Coerces raw into the right Python type for the SQLAlchemy column
        type.

        Args:
            column_type: The SQLAlchemy column type to coerce for.
            raw: The raw value to coerce.

        Returns:
            The coerced value, or None if the raw value is None.
        """

        if raw is None:
            return None

        # Fast path if already correct type
        if isinstance(column_type, (Integer, BigInteger, SmallInteger)):
            if isinstance(raw, int) and not isinstance(raw, bool):
                return raw
        elif isinstance(column_type, Float):
            if isinstance(raw, (int, float)) and not isinstance(raw, bool):
                return float(raw)

        # Normalize strings like "1,234.56" or "1.79E+11"
        if isinstance(raw, str):
            s = raw.strip().replace(',', '')
        else:
            s = str(raw)

        try:
            d = Decimal(s)
        except InvalidOperation as ex:
            raise ValueError(
                f'Could not parse numeric value from {raw!r}'
            ) from ex

        if isinstance(column_type, (Integer, BigInteger, SmallInteger)):
            if d != d.to_integral_value():
                raise ValueError(
                    'Expected an integer for this column, got a non-integer: '
                    f'{raw!r}'
                )
            return int(d)

        if isinstance(column_type, Float):
            return float(d)

        return raw
