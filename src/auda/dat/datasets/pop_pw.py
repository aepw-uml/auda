from datetime import date
from typing import List, override

from auda.core import DatabaseName
from auda.services.table import (
    TableQueryParams,
    TableService,
)
from auda.utils.pipeline import IOSpec, Task, task

from .__common import DATASET_KIND, DatasetISName, DatasetOSName, LabeledSamples, Units
from .__data_tables import (
    DataTableName,
    DemographyColumn,
    WasteGenerationManagementColumn,
)


@task(
    id='DS-POP-PW',
    kind=DATASET_KIND,
    description='Retrieves data linking population to plastic waste generation.',
    input_specs={
        DatasetISName.LOCATION: IOSpec(dtype=str, required=False),
        DatasetISName.YEAR: IOSpec(dtype=int, required=False),
    },
    output_specs={
        DatasetOSName.SAMPLES: IOSpec(dtype=LabeledSamples),
        DatasetOSName.ORIGINAL_SAMPLES: IOSpec(dtype=LabeledSamples),
        DatasetOSName.LABEL: IOSpec(dtype=str),
        DatasetOSName.FEATURE_NAMES: IOSpec(dtype=List[str]),
        DatasetOSName.UNITS: IOSpec(dtype=Units),
    },
)
class PopPwDataset(Task):
    @override
    def run(self) -> None:
        feature_columns = [
            DemographyColumn.POPULATION,
            WasteGenerationManagementColumn.PLASTIC_WASTE_GENERATED,
        ]

        table_service = TableService(DatabaseName.AUDA)
        table_result = table_service.prepare_get_table_result(
            [DataTableName.DEMOGRAPHY, DataTableName.WASTE_GENERATION_MANAGEMENT],
            TableQueryParams(
                column_names=[
                    DemographyColumn.LOCATION,
                    DemographyColumn.YEAR,
                    *feature_columns,
                ],
                notnull_column_names=feature_columns[:],
            ),
        )

        current_year = date.today().year
        rows = [row for row in table_result.data if row[1] <= current_year]

        # ---- Filter by location and year if provided
        location: str | None = self.get_input(DatasetISName.LOCATION)
        if location is not None:
            rows = [row for row in rows if row[0] == location]

        year: int | None = self.get_input(DatasetISName.YEAR)
        if year is not None:
            year = int(year)
            rows = [row for row in rows if row[1] == year]

        # ---- Construct samples from the rows
        samples: LabeledSamples = [(row[2:-1], row[3]) for row in rows]

        self.set_output(DatasetOSName.SAMPLES, samples)
        self.set_output(DatasetOSName.ORIGINAL_SAMPLES, samples)
        self.set_output(DatasetOSName.LABEL, 'Plastic Waste Generated (Metric Tons)')
        self.set_output(DatasetOSName.FEATURE_NAMES, ['Population'])
        self.set_output(DatasetOSName.UNITS, [None])
