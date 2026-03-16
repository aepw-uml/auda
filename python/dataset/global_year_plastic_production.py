import csv
from pathlib import Path
from typing import override

import numpy as np

from .dataset import Dataset, DatasetFetcher, DatasetSchema


class GlobalYearPlasticsProduction(DatasetFetcher):
    """Loads annual global plastics production from a local CSV file."""

    MIN_YEAR: int = 1960

    @override
    def fetch_dataset(self) -> Dataset:
        """Loads the dataset from ``data/global-plastics-production.csv``.

        Returns:
            The global annual plastics production dataset.

        Raises:
            ValueError: If the CSV file contains no usable world rows.
        """

        file_path = Path('data') / 'global-plastics-production.csv'
        samples: list[tuple[int, float]] = []

        with open(file_path, encoding='utf-8', newline='') as file:
            reader = csv.DictReader(file)
            for row in reader:
                entity = row.get('Entity', '')
                year_str = row.get('Year', '')
                production_str = row.get(
                    'Annual plastic production between 1950 and 2019', ''
                )

                if entity != 'World' or not year_str or not production_str:
                    continue

                year = int(year_str)
                if year < self.MIN_YEAR:
                    continue

                samples.append((year, float(production_str)))

        if not samples:
            raise ValueError(
                'No global plastics production rows were found in the CSV file.'
            )

        samples.sort(key=lambda sample: sample[0])
        years, production = zip(*samples)

        return Dataset(
            X=np.array([[year] for year in years], dtype=float),
            y=np.array(production, dtype=float),
        )

    @override
    def get_dataset_schema(self) -> DatasetSchema:
        """Returns the schema for the global plastics production dataset.

        Returns:
            The dataset schema.
        """

        return DatasetSchema(
            feature_names=['Year'],
            target_names=['Global Plastics Production'],
            feature_units=[''],
            target_units=['Tonnes'],
        )

    @override
    def fetch(self) -> tuple[Dataset, DatasetSchema]:
        """Loads and caches the global plastics production dataset.

        Returns:
            The dataset and its schema.
        """

        cache_key = f'GlobalYearPlasticsProduction?min_year={self.MIN_YEAR}'
        return super().fetch(cache_key)
