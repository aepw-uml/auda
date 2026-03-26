from dataclasses import dataclass


@dataclass(frozen=True)
class MigrationOptions:
    """Options for migrating data from the PRISM database to the AUDA database.

    Attributes:
        batch_size: The number of data points to process in each batch.
        is_validated: If True, only migrate data points that have been
            validated.
        allow_duplicates: If True, allow duplicate data points to be migrated.
        minimal_completeness_score: The minimum completeness score a data point
            must have to be migrated.
    """

    batch_size: int = 100
    is_validated: bool = True
    allow_duplicates: bool = True
    minimal_completeness_score: int = 0
