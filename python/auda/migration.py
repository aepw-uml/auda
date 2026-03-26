from typer import Option, Typer
from util.logging import get_logger
from util.migration import MigrationOptions
from util.migration.migration_service import MigrationService
from util.stopwatch import Stopwatch

app = Typer(name='migration', help='Manage migrations')


@app.command(
    name='start',
    help='Migrate data from the PRISM database to AEPW analysis tables.',
)
def start_migration(
    batch_size: int = Option(
        100, '--batch-size', help='Number of records to process in each batch.'
    ),
    is_validated: bool = Option(
        True, '--is-validated', help='Only migrate validated records.'
    ),
    minimal_completeness_score: int = Option(
        20,
        '--minimal-completeness-score',
        help='Minimal completeness score for records to be migrated.',
    ),
    minimal_data_source_score: int = Option(
        20,
        '--minimal-data-source-score',
        help='Minimal data source score for records to be migrated.',
    ),
) -> None:
    """
    Start the migration process.

    This command migrates data from the PRISM database to AEPW analysis tables.
    It processes records in batches and can filter based on validation status
    and completeness score.

    Args:
        batch_size (int): Number of records to process in each batch.
        is_validated (bool): Only migrate validated records if True.
        minimal_completeness_score (int): Minimal completeness score for records
            to be migrated.
    """

    migration_options = MigrationOptions(
        batch_size=batch_size,
        is_validated=is_validated,
        allow_duplicates=True,
        minimal_completeness_score=minimal_completeness_score,
        minimal_data_source_score=minimal_data_source_score,
    )

    stopwatch = Stopwatch().start()
    migration_service = MigrationService(migration_options)
    migration_service.migrate()
    stopwatch.stop()

    get_logger('MigrationCommand').info(
        f'Migration completed in {stopwatch.duration_seconds()} seconds.'
    )
