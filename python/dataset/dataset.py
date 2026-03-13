import hashlib
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import cast

import numpy as np
from util.cache import cache


@dataclass(frozen=True)
class DatasetSchema:
    """Schema for dataset.

    Attributes:
        feature_names: List of feature names.
        feature_units: List of feature units.
        target_names: List of target names.
        target_units: List of target units.
        class_names: List of class names (for classification models).
    """

    # Regression models
    feature_names: list[str]
    feature_units: list[str]
    target_names: list[str] | None = None
    target_units: list[str] | None = None

    # Classification models
    class_names: list[str] | None = None


@dataclass(frozen=True)
class Dataset:
    """Dataset containing features and labels.

    Attributes:
        X: The feature matrix (2D array).
        y: The label vector (1D array) or None if not available.
    """

    X: np.ndarray
    y: np.ndarray | None = None


class DatasetFetcher(ABC):
    @abstractmethod
    def fetch_dataset(self, *args, **kwargs) -> Dataset:
        """Fetches the dataset.

        Args:
            *args: Positional arguments.
            **kwargs: Keyword arguments.
        """

        pass

    @abstractmethod
    def get_dataset_schema(self) -> DatasetSchema:
        """Returns the schema of the dataset.

        This method can be overridden by subclasses to provide specific schema
        information. By default, it returns an empty schema.

        Returns:
            The schema of the dataset.
        """

        pass

    def fetch(
        self, cache_key: str, *args, **kwargs
    ) -> tuple[Dataset, DatasetSchema]:
        """Fetches the dataset using fetch_dataset and caches it immediately.

        This method first checks if the dataset is already cached on disk using
        the provided cache_key. If it is cached, it loads the dataset from disk.
        If not, it calls fetch_dataset to fetch the dataset, saves it to disk
        for future use, and then returns the dataset along with its schema.

        Args:
            cache_key: The key to use for caching the dataset.
            *args: Positional arguments to pass to fetch_dataset.
            **kwargs: Keyword arguments to pass to fetch_dataset.
        """

        dataset = self.get_dataset_from_cache(cache_key)
        if dataset is None:
            dataset = self.fetch_dataset(*args, **kwargs)
            self.save_dataset_to_cache(cache_key, dataset)

        return dataset, self.get_dataset_schema()

    def get_dataset_from_cache(self, cache_key: str) -> Dataset | None:
        """Retrieves the dataset from disk if it exists.

        Args:
            cache_key: The key to use for retrieving the dataset from cache.
        """

        path_str = cast(str | None, cache.get(cache_key))
        if not path_str:
            return None

        path = Path(path_str)
        if not path.exists():
            return None

        if path.suffix == '.npz':
            loaded = np.load(path)
            if 'X' not in loaded:
                return None

            return Dataset(
                X=loaded['X'],
                y=loaded['y'] if 'y' in loaded else None,
            )

        return None

    def save_dataset_to_cache(self, cache_key: str, dataset: Dataset) -> None:
        """Saves the dataset to disk.

        Args:
            cache_key: The key to use for saving the dataset to cache.
            dataset: The dataset to save.
        """

        hash_key = hashlib.sha256(cache_key.encode()).hexdigest()
        path = Path('cache') / f'{hash_key}.npz'
        cache[cache_key] = str(path)

        if dataset.y is None:
            np.savez(path, X=dataset.X)
        else:
            np.savez(path, X=dataset.X, y=dataset.y)
