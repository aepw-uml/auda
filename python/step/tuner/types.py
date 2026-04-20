from typing import Literal

from common.metrics import RegressionMetrics

Interval = tuple[float, float]
Hyperparameters = list[float]
Configuration = tuple[Hyperparameters, list[RegressionMetrics]]
Domain = list[Interval]
SearchSpace = list[Domain]
SamplingScale = Literal['uniform', 'log_uniform']

__all__ = [
    'Interval',
    'Hyperparameters',
    'Configuration',
    'Domain',
    'SearchSpace',
    'SamplingScale',
]
