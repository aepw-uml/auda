from datetime import timedelta
from time import time

from typer import Option, Typer

from auda.core import project
from auda.services.migration import MigrationOptions, MigrationService

app = Typer()


@app.command(name='start', help='Start the data migration process.')
def start_migration(
    max_count: int = Option(
        -1, '--max-count', help='Maximum number of records to migrate. -1 for no limit.'
    ),
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
) -> None:
    migration_service = project.singleton_registry.get(MigrationService)

    start_time = time()
    migration_service.migrate(
        MigrationOptions(
            max_count=max_count,
            batch_size=batch_size,
            is_validated=is_validated,
            minimal_completeness_score=minimal_completeness_score,
        )
    )
    duration_seconds = time() - start_time
    duration_seconds_str = str(timedelta(seconds=duration_seconds))
    project.get_logger('MigrationCommand').info(
        f'Migration completed in {duration_seconds_str} seconds.'
    )
