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


class PWGPredictors(DatasetFetcher):
    @override
    def fetch_dataset(self) -> Dataset:
        table_service = TableService(env.dbUrl)
        tables, table_metadata_map = table_service.prepare_tables(
            [
                DataTableName.DEMOGRAPHY,
                DataTableName.WASTE_GENERATION_MANAGEMENT,
            ]
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

        X, y = [], []
        for row in table_result.data:
            X.append(row[:-1])
            y.append(row[-1])

        for i in range(len(X)):
            X[i].append(X[i][3] / X[i][2])
            X[i].append(X[i][4] / X[i][2])
            X[i].append(X[i][0] / X[i][1])

        return Dataset(X=np.array(X), y=np.array(y))

    @override
    def get_dataset_schema(self) -> DatasetSchema:
        return DatasetSchema(
            feature_names=[
                'Year',
                'GDP',
                'Population',
                'Urban Population',
                'Rural Population',
                'Urban Population Ratio',
                'Rural Population Ratio',
                'GDP Per Capita',
            ],
            feature_units=['', 'US Dollars', '', '', '', '', '', '', ''],
            target_names=['Plastic Waste Generation'],
            target_units=['Metric Tonnes'],
        )

    @override
    def fetch(self, **kwargs) -> tuple[Dataset, DatasetSchema]:
        _ = kwargs
        cache_key = 'PWGPredictors'
        return super().fetch(cache_key)
