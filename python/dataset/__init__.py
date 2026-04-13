from common.dataset import DatasetFetcher

from .predicted_features import PredictedFeatures
from .year_ppc import YearPPC
from .year_pwg import YearPWG
from .year_trc import YearTRC

dataset_map: dict[str, type[DatasetFetcher]] = {
    'YearPWG': YearPWG,
    'YearTRC': YearTRC,
    'YearPPC': YearPPC,
    'PredictedFeatures': PredictedFeatures,
}

__all__ = ['YearPWG', 'YearTRC', 'YearPPC', 'PredictedFeatures', 'dataset_map']
