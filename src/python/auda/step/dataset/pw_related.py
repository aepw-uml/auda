from datetime import date
from typing import override

import numpy as np
from auda.core.constants import DatabaseName
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
    id='DS-PW-RELATED',
    description='Retrieves plastic waste generation data along with relevant '
    'demographic indicators.',
    output_specs=[Spec.DATASET, Spec.DATASET_SCHEMA],
)
class PwRelated(DatasetStep):
    @override
    def run(self) -> IOValueMap:
        cache_key = self.get_cache_key()
        dataset = self.fetch_and_cache_dataset(cache_key)

        return {
            Spec.DATASET.name: dataset,
            Spec.DATASET_SCHEMA.name: DatasetSchema(
                feature_names=[
                    'Year',
                    'GDP',
                    'Population',
                    'Urban Population',
                    'Rural Population',
                ],
                feature_units=[None, 'US Dollars', None, None, None],
                label_names=['Plastic Waste Generation'],
                label_units=['Tonnes'],
            ),
        }

    @override
    def fetch_dataset(self) -> Dataset:
        table_service = TableService(DatabaseName.AUDA)
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

        # ---- Construct samples from the table result
        X_vals, y_vals = [], []
        for row in table_result.data:
            # Note: the last column must be plastic waste generated
            X_vals.append(row[:-1])
            y_vals.append(row[-1])

        X = np.array(X_vals)
        y = np.array(y_vals)

        # ---- Filter out the samples where the year is greater than the
        # current year
        current_year = date.today().year
        correct_indices = np.where(X[:, 0] <= current_year)[0]
        X = X[correct_indices]
        y = y[correct_indices]

        return np.array(X), np.array(y)
