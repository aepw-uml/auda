from common.dataset import DatasetFetcher

from .predicted_features import PredictedFeatures
from .year_ppc import YearPPC

dataset_map: dict[str, type[DatasetFetcher]] = {
    'YearPPC': YearPPC,
    'PredictedFeatures': PredictedFeatures,
}

__all__ = ['PredictedFeatures', 'YearPPC', 'dataset_map']
