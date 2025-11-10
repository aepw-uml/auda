from decimal import Decimal, InvalidOperation
from logging import Logger
from math import ceil
from operator import and_
from typing import Any, Dict, List

from pydantic import BaseModel
from sqlalchemy import (
    BigInteger,
    Float,
    Integer,
    Select,
    SmallInteger,
    Table,
    func,
    insert,
    select,
    update,
)

from auda.core import Database, DatabaseName, project
from auda.models import (
    DataColumnMetadata,
    PrismDataExtraction,
    PrismDataPoint,
    PrismLocation,
)
from auda.services.data import DataService
from auda.services.prism import PrismService


class DataPoint(BaseModel):
    """
    Represents a data point in migration.
    """

    id: str
    data_point_id: str
    location_id: str
    year: int
    value: str


class MigrationOptions(BaseModel):
    """
    Options for migrating data from the AEPW database to the ADA database.

    Attributes:
        max_count: The maximum number of data points to migrate. A value of -1 means
            no limit. Default is -1.
        batch_size: The number of data points to process in each batch. Default is 100
        is_validated: If True, only migrate data points that have been validated.
            Default is True.
        allow_duplicates: If True, allow duplicate data points to be migrated.
        minimal_completeness_score: The minimum completeness score a data point must
            have to be migrated. Default is 0.
    """

    max_count: int = -1
    batch_size: int = 100
    is_validated: bool = True
    allow_duplicates: bool = True
    minimal_completeness_score: int = 0


class MigrationService:
    def __init__(self) -> None:
        self.logger: Logger = project.get_logger(self.__class__.__name__)
        self.prism_db: Database = project.database_manager.get(DatabaseName.PRISM)
        self.auda_db: Database = project.database_manager.get(DatabaseName.AUDA)
        self.prism_service: PrismService = project.singleton_registry.get(PrismService)
        self.data_service: DataService = project.singleton_registry.get(DataService)

    def migrate(self, options: MigrationOptions) -> None:
        """
        Migrates data points from the AEPW database to the ADA database based on the
        provided options.
        """

        def update_query(query: Select) -> Select:
            """
            Updates the given SQLAlchemy query based on the migration options.

            The `validation_status` can be either 'VALIDATED' or 'INVALIDATED'.

            This `data_completeness_score` is an integer between 0 and 100.
            """
            if options.is_validated:
                query = query.where(
                    PrismDataExtraction.validation_status == 'VALIDATED'
                )
            if not options.allow_duplicates:
                query = query.where(PrismDataExtraction.is_duplicated.is_(False))
            if options.minimal_completeness_score > 0:
                query = query.where(
                    PrismDataExtraction.data_completeness_score
                    >= options.minimal_completeness_score
                )

            return query

        self.logger.info('Migration started.')

        # Get the total number of records to migrate
        query = select(func.count()).select_from(PrismDataExtraction)
        query = update_query(query)
        num_records: int = 0
        with self.prism_db.get_session() as prism_session:
            num_records = prism_session.execute(query).scalar_one()
        self.logger.info(f'Found {num_records} data points in the PRISM database.')

        # Calculate the number of records to migrate
        num_records_to_migrate = (
            num_records
            if options.max_count == -1
            else max(0, min(num_records, options.max_count))
        )
        self.logger.info(f'Total records to migrate: {num_records_to_migrate}.')

        # Collect data column types from the `data_points` table in the Prism database
        self.logger.info('Collecting data column types...')
        prism_data_point_map: Dict[str, PrismDataPoint] = (
            self.prism_service.get_data_point_map()
        )
        self.logger.info(f'Collected {len(prism_data_point_map)} data column types.')

        # Collect locations from the `locations` table in the Prism database
        self.logger.info('Collecting locations from the `locations` table...')
        prism_location_map: Dict[str, PrismLocation] = (
            self.prism_service.get_location_map()
        )
        self.logger.info(f'Collected {len(prism_location_map)} locations.')

        # Collect data column metadata from the `data_column_metadata` table
        self.logger.info('Collecting data column metadata...')
        data_column_metadata_map: Dict[str, DataColumnMetadata] = (
            self.data_service.get_data_column_metadata_map()
        )
        self.logger.info(
            f'Collected {len(data_column_metadata_map)} pieces of data column metadata'
        )

        # Prepare data tables
        self.logger.info('Preparing data tables...')
        data_table_map: Dict[str, Table] = self.data_service.get_data_table_map()
        data_table_name_list: List[str] = list(data_table_map.keys())
        self.logger.info(f'Prepared {len(data_table_name_list)} data tables')

        # Select specific columns to avoid loading unnecessary data
        query = select(
            PrismDataExtraction.id,
            PrismDataExtraction.data_point_id,
            PrismDataExtraction.location_id,
            PrismDataExtraction.year,
            PrismDataExtraction.value,
        ).select_from(PrismDataExtraction)
        query = update_query(query)

        # Order the query by ID to ensure consistent ordering
        query = query.order_by(PrismDataExtraction.id.asc())

        # Calculate the number of batches
        batch_size: int = options.batch_size
        num_batches = ceil(num_records_to_migrate / batch_size)
        self.logger.info(f'The batch size is set to {batch_size}.')
        self.logger.info(f'Total batches to process: {num_batches}.')

        batch_index: int = 0
        migrated_count: int = 0
        skipped_count: int = 0
        inserted_count: int = 0

        while migrated_count < num_records_to_migrate:
            if migrated_count >= num_records_to_migrate:
                break

            self.logger.info(f'[{batch_index + 1}/{num_batches}] Migrating batch...')
            real_batch_size: int = min(
                num_records_to_migrate - migrated_count, batch_size
            )
            batch_query: Select = query.limit(real_batch_size).offset(
                batch_size * batch_index + batch_size
            )

            data_points: List[DataPoint] = []
            with self.prism_db.get_session() as prism_session:
                batch_data = prism_session.execute(batch_query).fetchall()
                data_points = [
                    DataPoint(
                        id=str(row.id),
                        data_point_id=str(row.data_point_id),
                        location_id=str(row.location_id),
                        year=row.year,
                        value=str(row.value),
                    )
                    for row in batch_data
                ]
            self.logger.debug(f'Fetched {len(data_points)} data points.')

            for data_point in data_points:
                inserted: bool = self.migrate_data_point(
                    data_point,
                    prism_data_point_map,
                    prism_location_map,
                    data_column_metadata_map,
                    data_table_map,
                )

                if inserted:
                    inserted_count += 1
                else:
                    skipped_count += 1

            batch_index += 1
            migrated_count += batch_size

        self.logger.info('Migration completed.')
        self.logger.info(f'Total data points migrated: {inserted_count}.')
        self.logger.info(f'Total data points skipped: {skipped_count}.')

    def migrate_data_point(
        self,
        data_point: DataPoint,
        prism_data_point_map: Dict[str, PrismDataPoint],
        prism_location_map: Dict[str, PrismLocation],
        data_column_metadata_map: Dict[str, DataColumnMetadata],
        data_table_map: Dict[str, Table],
    ) -> bool:
        data_point_id = data_point.data_point_id
        if data_point_id not in prism_data_point_map:
            self.logger.debug(
                f'Data point ID "{data_point_id}" not found in data column map; the '
                'data point will be skipped.'
            )
            return False

        aepw_data_point: PrismDataPoint = prism_data_point_map[data_point_id]
        original_column_name: str = aepw_data_point.name

        if original_column_name not in data_column_metadata_map:
            self.logger.debug(
                f'Original column name `{original_column_name}` not found in data '
                'column metadata map; the data point will be skipped.'
            )
            return False

        data_column_metadata: DataColumnMetadata = data_column_metadata_map[
            original_column_name
        ]
        data_table_name: str = data_column_metadata.table_name

        if data_table_name not in data_table_map:
            self.logger.debug(
                f'Data table name {data_table_name} not found in data table '
                'map; the data point will be skipped.'
            )
            return False

        data_table: Table = data_table_map[data_table_name]

        # The primary key (location and year)
        year: int = data_point.year

        # Get the location name
        location_id: str = data_point.location_id
        prism_location = prism_location_map.get(location_id)

        # The primary key (location, year)
        location: str = prism_location.location_name if prism_location else 'Unknown'
        year: int = data_point.year

        # Insert or update the data point
        self.insert_or_update_data_point(
            data_table,
            data_column_metadata.column_name,
            location,
            year,
            data_point.value,
        )

        return True

    def insert_or_update_data_point(
        self, data_table: Table, column_name: str, location: str, year: int, value: Any
    ) -> None:
        """
        Checks if the given data table contains a record whose primary key is
        (location, year). If it does, update the value of the specified column. If not,
        insert a new record, settings all other columns to NULL.
        """
        # Assume that the target column exists on the data table
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

        # The primary key of all data tables is (location, year)
        where_pk = and_(location_col == location, year_col == year)

        with self.auda_db.get_session() as ada_db_session:
            with ada_db_session.begin():
                # First, try to update an existing record
                upd_stmt = (
                    update(data_table).where(where_pk).values({target_col.key: value})
                )
                result = ada_db_session.execute(upd_stmt)

                # If no record was updated, insert a new one
                if result.rowcount == 0:
                    insert_values = {
                        location_col.key: location,
                        year_col.key: year,
                        target_col.key: value,
                    }
                    insert_stmt = insert(data_table).values(insert_values)
                    ada_db_session.execute(insert_stmt)

    def _coerce_for_column(self, column_type, raw):
        """Coerce raw into the right Python type for the SQLAlchemy column type."""
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
            raise ValueError(f'Could not parse numeric value from {raw!r}') from ex

        if isinstance(column_type, (Integer, BigInteger, SmallInteger)):
            if d != d.to_integral_value():
                raise ValueError(
                    f'Expected an integer for this column, got a non-integer: {raw!r}'
                )
            return int(d)

        if isinstance(column_type, Float):
            return float(d)

        return raw
