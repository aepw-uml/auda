from common.dataset import DatasetFetcher

from .global_plastics_production import GlobalPlasticsProduction
from .pw_driver_feature_set import PWDriverFeatureSet
from .pw_drivers import PWDrivers
from .pwg_predictors import PWGPredictors
from .year_ppc import YearPPC
from .year_pwg import YearPWG
from .year_trc import YearTRC

dataset_map: dict[str, type[DatasetFetcher]] = {
    'YearPWG': YearPWG,
    'YearTRC': YearTRC,
    'YearPPC': YearPPC,
    'PWDriverFeatureSet': PWDriverFeatureSet,
    'PWGPredictors': PWGPredictors,
    'PWDrivers': PWDrivers,
    'GlobalPlasticsProduction': GlobalPlasticsProduction,
}

__all__ = [
    'YearPWG',
    'YearTRC',
    'YearPPC',
    'PWDriverFeatureSet',
    'PWGPredictors',
    'PWDrivers',
    'GlobalPlasticsProduction',
    'dataset_map',
]
