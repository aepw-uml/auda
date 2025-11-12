from typing import List, override

from auda.core import DatabaseName
from auda.dat.datasets.__common import DatasetOSName, UnlabeledSample, UnlabeledSamples
from auda.services.table import TableQueryParams, TableService
from auda.utils.pipeline import IOSpec, Task, task

from .__common import DATASET_KIND, LabeledSamples, Units
from .__data_tables import (
    DataTableName,
    DemographyColumn,
    WasteGenerationManagementColumn,
)


@task(
    id='DS-SIGNIFICANT-FEATURES',
    kind=DATASET_KIND,
    description='Collects samples containing key demographic and waste management '
    'indicators.',
    output_specs={
        DatasetOSName.SAMPLES: IOSpec(dtype=UnlabeledSamples),
        DatasetOSName.ORIGINAL_SAMPLES: IOSpec(dtype=LabeledSamples),
        DatasetOSName.LABEL: IOSpec(dtype=str),
        DatasetOSName.FEATURE_NAMES: IOSpec(dtype=List[str]),
        DatasetOSName.UNITS: IOSpec(dtype=Units),
    },
)
class SignificantFeaturesDataset(Task):
    @override
    def run(self) -> None:
        column_names: List[str] = [
            DemographyColumn.POPULATION,
            DemographyColumn.GDP,
            WasteGenerationManagementColumn.PLASTIC_WASTE_GENERATED,
            WasteGenerationManagementColumn.PLASTIC_WASTE_GENERATED_PER_CAPITA,
            WasteGenerationManagementColumn.PLASTIC_MISMANAGED,
            WasteGenerationManagementColumn.PLASTIC_MISMANAGED_PERCENT,
            WasteGenerationManagementColumn.PLASTIC_WASTE_COLLECTED_PERCENT,
        ]

        table_service = TableService(DatabaseName.AUDA)
        tables, table_metadata_map = table_service.prepare_tables(
            [DataTableName.DEMOGRAPHY, DataTableName.WASTE_GENERATION_MANAGEMENT]
        )

        table_result = table_service.get_table_result(
            tables,
            table_metadata_map,
            TableQueryParams(
                column_names=column_names,
                notnull_column_names=column_names,
            ),
        )

        # ---- Construct unlabeled samples
        samples: UnlabeledSamples = []
        for row in table_result.data:
            sample: UnlabeledSample = []
            for i in range(len(row)):
                sample.append(row[i])

            samples.append(sample)

        self.set_output(DatasetOSName.SAMPLES, samples)
        self.set_output(DatasetOSName.ORIGINAL_SAMPLES, samples)
        self.set_output(
            DatasetOSName.FEATURE_NAMES,
            [
                'Population',
                'GDP',
                'Plastic Waste Generated',
                'Plastic Waste Generated Per Capita',
                'Plastic Mismanaged',
                'Plastic Mismanaged Percent',
                'Plastic Waste Collected Percent',
            ],
        )
        self.set_output(DatasetOSName.LABEL, 'Significant Features')
        self.set_output(
            DatasetOSName.UNITS,
            [
                None,
                'US Dollars',
                'Metric Tons',
                'Kg/Person/Year',
                'Metric Tons',
                None,
                None,
            ],
        )
