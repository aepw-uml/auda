from typing import List, override

import numpy as np
from auda.core import DatabaseName
from auda.service.table_service import TableQueryParams, TableService
from auda.step.dataset import DatasetSchema, DatasetStep
from auda.step.spec import Dataset, Spec
from auda.utils.pipeline import IOValueMap, step

from .__data_tables import (
    DataTableName,
    DemographyColumn,
    WasteGenerationManagementColumn,
)


@step(
    id='DS-MAJOR-FEATURES',
    description='Collects samples containing key demographic and waste '
    'management indicators.',
    input_specs=[],
    output_specs=[Spec.DATASET, Spec.DATASET_SCHEMA],
)
class SignificantFeaturesDataset(DatasetStep):
    @override
    def run(self) -> IOValueMap:
        cache_key = self.get_cache_key()
        dataset = self.fetch_and_cache_dataset(cache_key)

        return {
            Spec.DATASET.name: dataset,
            Spec.DATASET_SCHEMA.name: DatasetSchema(
                feature_names=[
                    'Population',
                    'GDP',
                    'Plastic Waste Generated',
                    'Plastic Waste Generated Per Capita',
                    'Plastic Mismanaged',
                    'Plastic Mismanaged Percent',
                    'Plastic Waste Collected Percent',
                ],
                feature_units=[
                    None,
                    'US Dollars',
                    'Metric Tons',
                    'Kg/Person/Year',
                    'Metric Tons',
                    None,
                    None,
                ],
            ),
        }

    @override
    def fetch_dataset(self) -> Dataset:
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
            [
                DataTableName.DEMOGRAPHY,
                DataTableName.WASTE_GENERATION_MANAGEMENT,
            ]
        )

        table_result = table_service.get_table_result(
            tables,
            table_metadata_map,
            TableQueryParams(
                column_names=column_names,
                notnull_column_names=column_names,
            ),
        )

        return np.array(table_result.data)
