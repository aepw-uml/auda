from datetime import date
from typing import override

import numpy as np
from auda.core import DatabaseName
from auda.service.table_service import TableQueryParams, TableService
from auda.step.dataset import DatasetSchema, DatasetStep
from auda.step.spec import Dataset, Spec
from auda.utils.pipeline import IOValueMap, step

from .__data_tables import DataTableName, WasteGenerationManagementColumn


@step(
    id='DS-YEAR-PW',
    description='Retrieves a dataset containing yearly plastic waste '
    'generation statistics.',
    input_specs=[Spec.LOCATION],
    output_specs=[Spec.DATASET, Spec.DATASET_SCHEMA],
)
class YearPlasticWaste(DatasetStep):
    @override
    def run(self, location: str) -> IOValueMap:
        cache_key = self.get_cache_key(location=location)
        dataset = self.fetch_and_cache_dataset(cache_key, location)

        return {
            Spec.DATASET.name: dataset,
            Spec.DATASET_SCHEMA.name: DatasetSchema(
                feature_names=['Year'],
                label_names=['Plastic Waste Generated'],
                feature_units=[''],
                label_units=['Tonnes'],
            ),
        }

    @override
    def fetch_dataset(self, location: str) -> Dataset:
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
            lambda q: q.where(
                waste_generation_management.c.location == location
            ),
        )
        result = table_result.data

        # ---- Process data
        current_year = date.today().year
        pairs = [
            (year, waste) for (year, waste) in result if year <= current_year
        ]

        if not pairs:
            raise ValueError(
                f'No plastic waste data found for location: {location}'
            )

        years, wastes = zip(*pairs)
        X = np.array([[year] for year in years], dtype=float)
        y = np.array(wastes, dtype=float)

        return (X, y)
