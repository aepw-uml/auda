from datetime import date
from typing import override

import numpy as np
from auda.core import DatabaseName
from auda.service.table_service import TableQueryParams, TableService
from auda.step.dataset import DatasetSchema, DatasetStep
from auda.step.dataset.__data_tables import (
    DataTableName,
    DemographyColumn,
    WasteGenerationManagementColumn,
)
from auda.step.spec import Dataset, Spec
from auda.utils.pipeline import IOValueMap, step


@step(
    id='DS-GPD-URBN-POP-PW',
    description='',
    input_specs=[Spec.LOCATION.optional(), Spec.YEAR.optional()],
    output_specs=[Spec.DATASET, Spec.DATASET_SCHEMA],
)
class GdpUrbanPopPw(DatasetStep):
    @override
    def run(self, location: str | None, year: int | None) -> IOValueMap:
        cache_key = self.get_cache_key(location=location, year=year)
        dataset = self.fetch_and_cache_dataset(cache_key, location, year)

        return {
            Spec.DATASET.name: dataset,
            Spec.DATASET_SCHEMA.name: DatasetSchema(
                feature_names=['GDP', 'Urban Population'],
                feature_units=['US Dollars', None],
                label_names=['Plastic Waste Generation'],
                label_units=['Tonnes'],
            ),
        }

    @override
    def fetch_dataset(self, location: str | None, year: int | None) -> Dataset:
        feature_columns = [
            DemographyColumn.GDP,
            DemographyColumn.URBAN_POPULATION,
            WasteGenerationManagementColumn.PLASTIC_WASTE_GENERATED,
        ]

        table_service = TableService(DatabaseName.AUDA)
        table_result = table_service.prepare_get_table_result(
            [
                DataTableName.DEMOGRAPHY,
                DataTableName.WASTE_GENERATION_MANAGEMENT,
            ],
            TableQueryParams(
                column_names=[
                    DemographyColumn.LOCATION,
                    DemographyColumn.YEAR,
                    *feature_columns,
                ],
                notnull_column_names=feature_columns[:],
            ),
        )
        # Column sequence:
        # [location, year, gdp, urban_population, plastic_waste_generated]

        current_year = date.today().year
        rows = [row for row in table_result.data if row[1] <= current_year]

        # ---- Filter by location and year if provided
        if location is not None:
            rows = [row for row in rows if row[0] == location]

        if year is not None:
            rows = [row for row in rows if row[1] == year]

        X = np.array([row[2:-1] for row in rows])
        y = np.array([row[-1] for row in rows])

        return X, y
