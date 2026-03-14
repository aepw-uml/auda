from datetime import date
from typing import override

import numpy as np
from util.database import TableQueryParams, TableService
from util.env import env

from .data_tables import DataTableName, PlasticResinColumn
from .dataset import Dataset, DatasetFetcher, DatasetSchema


class YearTRC(DatasetFetcher):
    @override
    def fetch_dataset(self, location: str) -> Dataset:
        table_service = TableService(env.dbUrl)
        tables, table_metadata_map = table_service.prepare_tables(
            [DataTableName.PLASTIC_RESIN]
        )
        waste_generation_management = tables[0]

        table_result = table_service.get_table_result(
            tables,
            table_metadata_map,
            TableQueryParams(
                column_names=[
                    PlasticResinColumn.YEAR,
                    PlasticResinColumn.TOTAL_RESIN_CONSUMPTION,
                ],
                notnull_column_names=[
                    PlasticResinColumn.TOTAL_RESIN_CONSUMPTION,
                ],
            ),
            lambda q: q.where(
                waste_generation_management.c.location == location
            ),
        )
        result = table_result.data

        current_year = date.today().year
        valid_samples = [
            (year, waste) for year, waste in result if year <= current_year
        ]

        if not valid_samples:
            raise ValueError(
                f'No total resin consumption found for location: {location}'
            )

        sorted_samples = sorted(valid_samples, key=lambda sample: sample[0])

        years, wastes = zip(*sorted_samples)
        X = np.array([[year] for year in years], dtype=float)
        y = np.array(wastes, dtype=float)

        return Dataset(X, y)

    @override
    def get_dataset_schema(self) -> DatasetSchema:
        return DatasetSchema(
            feature_names=['Year'],
            target_names=['Total Resin Consumption'],
            feature_units=[''],
            target_units=['Tonnes'],
        )

    @override
    def fetch(self, location: str) -> tuple[Dataset, DatasetSchema]:
        cache_key = f'YearTRC?location={location}'
        return super().fetch(cache_key, location)
