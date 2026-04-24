from typing import override

import numpy as np
from common.dataset import Dataset, DatasetFetcher, DatasetSchema
from util.database import TableQueryParams, TableService
from util.env import env

from .tables import (
    DataTableName,
    DemographyColumn,
    WasteGenerationManagementColumn,
)


class PWDriverFeatureSet(DatasetFetcher):
    @override
    def fetch_dataset(self) -> Dataset:
        table_service = TableService(env.dbUrl)
        column_names: list[str] = [
            DemographyColumn.POPULATION,
            DemographyColumn.GDP,
            DemographyColumn.URBAN_POPULATION,
            WasteGenerationManagementColumn.PLASTIC_WASTE_GENERATED,
            WasteGenerationManagementColumn.PLASTIC_MISMANAGED,
            WasteGenerationManagementColumn.PLASTIC_WASTE_COLLECTED_PERCENT,
        ]
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

        # ---- Construct unlabeled samples
        samples = []
        for row in table_result.data:
            sample = []
            for i in range(len(row)):
                sample.append(row[i])

            samples.append(sample)

        return Dataset(X=np.array(samples), y=None)

    @override
    def get_dataset_schema(self) -> DatasetSchema:
        return DatasetSchema(
            feature_names=[
                'Population',
                'GDP',
                'Urban Population',
                'Plastic Waste Generation',
                'Plastic Mismanaged',
                'Plastic Generation Management %',
            ],
            feature_units=[
                '',
                'US Dollars',
                '',
                'Metric Tons',
                'Metric Tons',
                '',
            ],
        )

    @override
    def fetch(self, **kwargs) -> tuple[Dataset, DatasetSchema]:
        _ = kwargs
        cache_key = 'PWDriverFeatureSet'
        return super().fetch(cache_key)
