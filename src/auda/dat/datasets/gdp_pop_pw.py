from datetime import date
from typing import List, override

from auda.core import DatabaseName
from auda.services.table import (
    TableQueryParams,
    TableService,
)
from auda.utils.pipeline import IOSpec, Task, task

from .__common import DATASET_KIND, DatasetOSName, LabeledSamples, Units
from .__data_tables import (
    DataTableName,
    DemographyColumn,
    WasteGenerationManagementColumn,
)


@task(
    id='DS-PW-RELATED',
    kind=DATASET_KIND,
    description='Retrieves plastic waste generation data along with relevant '
    'demographic indicators.',
    output_specs={
        DatasetOSName.SAMPLES: IOSpec(dtype=LabeledSamples),
        DatasetOSName.ORIGINAL_SAMPLES: IOSpec(dtype=LabeledSamples),
        DatasetOSName.LABEL: IOSpec(dtype=str),
        DatasetOSName.FEATURE_NAMES: IOSpec(dtype=List[str]),
        DatasetOSName.UNITS: IOSpec(dtype=Units),
    },
)
class PwRelatedDataset(Task):
    @override
    def run(self) -> None:
        table_service = TableService(DatabaseName.AUDA)
        tables, table_metadata_map = table_service.prepare_tables(
            [DataTableName.DEMOGRAPHY, DataTableName.WASTE_GENERATION_MANAGEMENT]
        )

        featured_columns = [
            DemographyColumn.GDP,
            DemographyColumn.POPULATION,
            DemographyColumn.URBAN_POPULATION,
            DemographyColumn.RURAL_POPULATION,
            WasteGenerationManagementColumn.PLASTIC_WASTE_GENERATED,
        ]

        table_result = table_service.get_table_result(
            tables,
            table_metadata_map,
            TableQueryParams(
                column_names=[
                    DemographyColumn.YEAR,
                    *featured_columns,
                ],
                notnull_column_names=featured_columns[:],
            ),
        )

        # ---- Construct samples from the table result
        samples: LabeledSamples = []
        for row in table_result.data:
            samples.append((row[:-1], row[5]))

        # ---- Order the samples by years
        samples.sort(key=lambda sample: sample[0][0])

        # ---- Filter out the samples where the year is greater than the current year
        # These samples may come from some prediction reports
        current_year = date.today().year
        samples = [sample for sample in samples if sample[0][0] <= current_year]

        self.set_output(DatasetOSName.SAMPLES, samples)
        self.set_output(DatasetOSName.ORIGINAL_SAMPLES, samples)
        self.set_output(DatasetOSName.LABEL, 'Plastic Waste Generated (Metric Tons)')
        self.set_output(
            DatasetOSName.FEATURE_NAMES,
            ['Year', 'GDP', 'Population', 'Urban Population', 'Rural Population'],
        )
        self.set_output(DatasetOSName.UNITS, [None, 'US Dollars', None, None, None])
