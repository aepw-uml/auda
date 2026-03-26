from dataclasses import dataclass


@dataclass(frozen=True)
class Datapoint:
    """Represents a datapoint in the migration process."""

    id: str
    datapoint_id: str
    location_id: str
    year: int
    value: str
    unit: str
