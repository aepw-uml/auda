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
    WasteGenerationManagementColumn,
)


@task(
    id='DS-YEAR-PW',
    kind=DATASET_KIND,
    description='Retrieves yearly plastic waste generation data for a specific '
    'location.',
    input_specs={
        DatasetISName.LOCATION: IOSpec(dtype=str),
    },
    output_specs={
        DatasetOSName.SAMPLES: IOSpec(dtype=LabeledSamples),
        DatasetOSName.ORIGINAL_SAMPLES: IOSpec(dtype=LabeledSamples),
        DatasetOSName.LABEL: IOSpec(dtype=str),
        DatasetOSName.FEATURE_NAMES: IOSpec(dtype=List[str]),
        DatasetOSName.UNITS: IOSpec(dtype=Units),
    },
)
class YearPw(Task):
    """
    Task to retrieve plastic waste generation data for a specific location.

    This task fetches data from the "d_waste_generation_management" data table, and
    filters it based on the provided location. The output includes labeled samples
    with the following data structure:

        ([year], plastic_waste_generated)
    """

    @override
    def run(self) -> None:
        location = self.get_input(DatasetISName.LOCATION)

        table_service = TableService(DatabaseName.AUDA)
        tables, table_metadata_map = table_service.prepare_tables(
            [DataTableName.WASTE_GENERATION_MANAGEMENT]
        )
        waste_generation_management = tables[0]

        table_result = table_service.get_table_result(
            tables,
            table_metadata_map,
            TableQueryParams(
                column_names=[
                    WasteGenerationManagementColumn.YEAR,
                    WasteGenerationManagementColumn.PLASTIC_WASTE_GENERATED,
                ],
                notnull_column_names=[
                    WasteGenerationManagementColumn.PLASTIC_WASTE_GENERATED,
                ],
            ),
            lambda q: q.where(waste_generation_management.c.location == location),
        )

        # ---- Construct labeled samples from the table result
        samples: LabeledSamples = []
        for row in table_result.data:
            if row[1] is not None:
                samples.append(([row[0]], row[1]))

        # ---- Sort samples by year
        samples.sort(key=lambda sample: sample[0][0])

        # ---- Filter out future years
        current_year = date.today().year
        samples = [sample for sample in samples if sample[0][0] <= current_year]

        if not samples:
            raise ValueError(f'No plastic waste data found for location: {location}')

        self.set_output(DatasetOSName.SAMPLES, samples)
        self.set_output(DatasetOSName.ORIGINAL_SAMPLES, samples)
        self.set_output(DatasetOSName.LABEL, 'Plastic Waste Generated (Metric Tons)')
        self.set_output(DatasetOSName.FEATURE_NAMES, ['Year'])
        self.set_output(DatasetOSName.UNITS, [None, 'Metric Tons'])
