from common.dataset import DatasetFetcher

from .global_year_plastic_production import GlobalYearPlasticsProduction
from .plastic_waste_driver_feature_set import PlasticWasteDriverFeatureSet
from .plastic_waste_generation_predictors import (
    PlasticWasteGenerationPredictors,
)
from .plastic_waste_drivers import PlasticWasteDrivers
from .year_ppc import YearPPC
from .year_pwg import YearPWG
from .year_trc import YearTRC

dataset_map: dict[str, type[DatasetFetcher]] = {
    'YearPWG': YearPWG,
    'YearTRC': YearTRC,
    'YearPPC': YearPPC,
    'PlasticWasteDriverFeatureSet': PlasticWasteDriverFeatureSet,
    'PlasticWasteGenerationPredictors': PlasticWasteGenerationPredictors,
    'PlasticWasteDrivers': PlasticWasteDrivers,
    'GlobalYearPlasticsProduction': GlobalYearPlasticsProduction,
}

__all__ = [
    'YearPWG',
    'YearTRC',
    'YearPPC',
    'PlasticWasteDriverFeatureSet',
    'PlasticWasteGenerationPredictors',
    'PlasticWasteDrivers',
    'GlobalYearPlasticsProduction',
    'dataset_map',
]
