from datetime import date
from typing import override

import numpy as np
from auda.core import DatabaseName
from auda.service.table_service import TableQueryParams, TableService
from auda.step.spec import Spec, SpecName
from auda.utils.pipeline import IOValueMap, Step, step

from .__data_tables import DataTableName, WasteGenerationManagementColumn


@step(
    id='DS-YEAR-PW',
    description='Retrieves a dataset containing yearly plastic waste '
    'generation statistics.',
    input_specs=[Spec.LOCATION],
    output_specs=[
        Spec.DATASET,
        Spec.FEATURE_NAMES,
        Spec.LABEL_NAMES,
        Spec.FEATURE_UNITS,
        Spec.LABEL_UNITS,
    ],
)
class YearPlasticWaste(Step):
    @override
    def run(self, location: str) -> IOValueMap:
        # ---- Fetch data from the database
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
        features = np.array([[y] for y in years], dtype=float)
        labels = np.array(wastes, dtype=float)

        return {
            SpecName.DATASET: (features, labels),
            SpecName.FEATURE_NAMES: ['Year'],
            SpecName.LABEL_NAMES: ['Plastic Waste Generated'],
            SpecName.FEATURE_UNITS: [''],
            SpecName.LABEL_UNITS: ['Tonnes'],
        }
