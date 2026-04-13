from datetime import date
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


class PredictedFeatures(DatasetFetcher):
    @override
    def fetch_dataset(self) -> Dataset:
        table_service = TableService(env.dbUrl)
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
                column_names=[
                    DemographyColumn.YEAR,
                    DemographyColumn.URBAN_POPULATION,
                    DemographyColumn.GDP,
                    WasteGenerationManagementColumn.PLASTIC_WASTE_GENERATED,
                ],
                notnull_column_names=[
                    DemographyColumn.URBAN_POPULATION,
                    DemographyColumn.GDP,
                    WasteGenerationManagementColumn.PLASTIC_WASTE_GENERATED,
                ],
            ),
        )
        result = table_result.data

        current_year = date.today().year
        valid_samples = [
            [year, urbun_population, gdp, plastic_waste_generation]
            for year, urbun_population, gdp, plastic_waste_generation in result
            if year <= current_year
        ]

        X = np.array([sample[:3] for sample in valid_samples], dtype=float)
        y = np.array([sample[3] for sample in valid_samples], dtype=float)

        return Dataset(X, y)

    @override
    def get_dataset_schema(self) -> DatasetSchema:
        return DatasetSchema(
            feature_names=['Year', 'Urban Population', 'GDP'],
            target_names=['Plastic Waste Generation'],
            feature_units=['', '', 'USD Dollars'],
            target_units=['Tonnes'],
        )

    @override
    def fetch(self, **kwargs) -> tuple[Dataset, DatasetSchema]:
        _ = kwargs
        cache_key = 'SignificantFeatures'
        return super().fetch(cache_key)
