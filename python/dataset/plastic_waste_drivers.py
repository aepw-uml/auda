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


class PWDrivers(DatasetFetcher):
    @override
    def fetch_dataset(self, location: str, exclude_location: str) -> Dataset:
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
                    DemographyColumn.LOCATION,
                    DemographyColumn.YEAR,
                    DemographyColumn.RURAL_POPULATION,
                    DemographyColumn.URBAN_POPULATION,
                    DemographyColumn.GDP,
                    WasteGenerationManagementColumn.PLASTIC_WASTE_GENERATED,
                ],
                notnull_column_names=[
                    DemographyColumn.RURAL_POPULATION,
                    DemographyColumn.GDP,
                    WasteGenerationManagementColumn.PLASTIC_WASTE_GENERATED,
                ],
            ),
        )
        result = table_result.data

        current_year = date.today().year
        valid_samples = [
            [location, year, rural_pop, urbun_pop, gdp, pwg]
            for location, year, rural_pop, urbun_pop, gdp, pwg in result
            if year <= current_year
        ]

        X = np.array([sample[1:4] for sample in valid_samples], dtype=float)
        y = np.array([sample[4] for sample in valid_samples], dtype=float)

        if location:
            location_samples = [
                sample for sample in valid_samples if sample[0] == location
            ]
            if not location_samples:
                raise ValueError(f'No data found for location: {location}')
            X = np.array(
                [sample[1:4] for sample in location_samples], dtype=float
            )
            y = np.array(
                [sample[4] for sample in location_samples], dtype=float
            )

        if exclude_location:
            X = np.array(
                [
                    sample[1:4]
                    for sample in valid_samples
                    if sample[0] != exclude_location
                ],
                dtype=float,
            )
            y = np.array(
                [
                    sample[4]
                    for sample in valid_samples
                    if sample[0] != exclude_location
                ],
                dtype=float,
            )

        return Dataset(X, y)

    @override
    def get_dataset_schema(self) -> DatasetSchema:
        return DatasetSchema(
            feature_names=[
                'Year',
                'Rural Population',
                'Urban Population',
                'GDP',
            ],
            feature_units=['', '', '', 'USD Dollars'],
            target_names=['Plastic Waste Generation'],
            target_units=['Tonnes'],
        )

    @override
    def fetch(
        self, location: str = '', exclude_location: str = '', **kwargs
    ) -> tuple[Dataset, DatasetSchema]:
        _ = kwargs
        cache_key = (
            'PWDrivers'
            + (f'?location={location}' if location else '')
            + (
                f'&exclude_location={exclude_location}'
                if exclude_location
                else ''
            )
        )
        return super().fetch(cache_key, location, exclude_location)
