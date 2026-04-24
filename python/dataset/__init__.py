from common.dataset import DatasetFetcher

from .global_year_plastic_production import GlobalPlasticsProduction
from .plastic_waste_driver_feature_set import PWDriverFeatureSet
from .plastic_waste_drivers import PWDrivers
from .plastic_waste_generation_predictors import PWGPredictors
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
