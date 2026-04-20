from typing import Literal

from common.metrics import RegressionMetrics

Interval = tuple[float, float]
Hyperparameters = list[float]
Trial = tuple[Hyperparameters, list[RegressionMetrics]]
Domain = list[Interval]
SearchSpace = list[Domain]
SamplingScale = Literal['uniform', 'log_uniform']

__all__ = [
    'Interval',
    'Hyperparameters',
    'Trial',
    'Domain',
    'SearchSpace',
    'SamplingScale',
]
