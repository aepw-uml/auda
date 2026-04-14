from typing import Literal

from common.metrics import RegressionMetrics

Interval = tuple[float, float]
Hyperparameters = list[float]
HyperparameterScore = tuple[float, Hyperparameters, list[RegressionMetrics]]
Domain = list[Interval]
SearchSpace = list[Domain]
SamplingScale = Literal['uniform', 'log_uniform']

__all__ = [
    'Interval',
    'Hyperparameters',
    'HyperparameterScore',
    'Domain',
    'SearchSpace',
    'SamplingScale',
]
