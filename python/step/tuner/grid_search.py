from itertools import product
from logging import Logger
from typing import Callable, Literal

import numpy as np
from common.metrics import RegressionMetricName, RegressionMetrics

Interval = tuple[float, float]
Hyperparameters = list[float]
HyperparameterScore = tuple[float, Hyperparameters]
Domain = list[Interval]
SearchSpace = list[Domain]
SamplingScale = Literal['uniform', 'log_uniform']


def grid_search(
    hyperparameter_names: list[str],
    search_space: SearchSpace,
    evaluate_hyperparameters: Callable[[Hyperparameters], RegressionMetrics],
    sampling_scales: list[SamplingScale],
    metric: RegressionMetricName = 'mape',
    num_points_per_interval: int = 5,
    logger: Logger | None = None,
) -> list[HyperparameterScore]:
    """Enumerates and evaluates a deterministic hyperparameter grid.

    For each hyperparameter, this function generates
    ``num_points_per_interval`` evenly spaced values inside every interval in
    that hyperparameter's domain. The per-hyperparameter value lists are then
    combined with a Cartesian product, and each resulting hyperparameter vector
    is passed to ``evaluate_hyperparameters``. The requested metric value is
    stored together with the evaluated hyperparameters. The function returns
    all ``(score, hyperparameters)`` pairs in evaluation order and does not
    sort or rank them.

    Args:
        hyperparameter_names: A list of hyperparameter names.
        search_space: A list of lists of intervals, where each inner list
            corresponds to a hyperparameter and contains the intervals from
            which to build grid values.
        evaluate_hyperparameters: A function that takes a list of hyperparameter
            values and returns a RegressionMetrics object containing the
            evaluation metrics for those hyperparameters.
        sampling_scales: A list of sampling scales for each hyperparameter.
        metric: The name of the metric to optimize.
        num_points_per_interval: The number of grid values to generate inside
            each interval.
        logger: An optional logger to log the progress of the grid search.

    Returns:
        A list of ``(score, hyperparameters)`` tuples in the order they were
        evaluated.
    """

    if len(search_space) != len(hyperparameter_names):
        raise ValueError(
            'Length of search_space must match length of hyperparameter_names.'
        )

    if len(sampling_scales) != len(hyperparameter_names):
        raise ValueError(
            'Length of sampling_scales must match length of '
            'hyperparameter_names.'
        )

    if num_points_per_interval < 1:
        raise ValueError('num_points_per_interval must be at least 1.')

    hyperparameter_grids: list[Hyperparameters] = [
        build_hyperparameter_grid(
            intervals=intervals,
            sampling_scale=sampling_scales[i],
            num_points_per_interval=num_points_per_interval,
        )
        for i, intervals in enumerate(search_space)
    ]

    total_iterations = int(
        np.prod([len(grid) for grid in hyperparameter_grids], dtype=int)
    )
    hyperparameter_scores: list[HyperparameterScore] = []
    for i, hyperparameters_tuple in enumerate(product(*hyperparameter_grids)):
        hyperparameters = list(hyperparameters_tuple)
        metrics: RegressionMetrics = evaluate_hyperparameters(hyperparameters)
        score: float = metrics.get_value_by_name(metric)
        hyperparameter_scores.append((score, hyperparameters))

        if logger is not None:
            hyperparameters_str = ', '.join(
                [
                    f'{name}={value:.3e}'
                    for name, value in zip(
                        hyperparameter_names, hyperparameters
                    )
                ]
            )
            logger.info(
                f'[{i + 1}/{total_iterations}] '
                f'Hyperparameters = ({hyperparameters_str}); '
                f'score = {score:.3e}'
            )

    return hyperparameter_scores


def build_hyperparameter_grid(
    intervals: Domain,
    sampling_scale: SamplingScale,
    num_points_per_interval: int,
) -> Hyperparameters:
    """Builds the grid values for one hyperparameter.

    Args:
        intervals: A list of intervals from which to generate grid values.
        sampling_scale: The sampling scale to use.
        num_points_per_interval: The number of values to generate inside each
            interval.

    Returns:
        A list of deterministic grid values for the hyperparameter.
    """

    values: Hyperparameters = []
    seen_values: set[float] = set()
    for low, high in intervals:
        interval_values = build_interval_grid(
            low=low,
            high=high,
            sampling_scale=sampling_scale,
            num_points=num_points_per_interval,
        )
        for value in interval_values:
            if value in seen_values:
                continue

            seen_values.add(value)
            values.append(value)

    return values


def build_interval_grid(
    low: float,
    high: float,
    sampling_scale: SamplingScale,
    num_points: int,
) -> Hyperparameters:
    """Builds evenly spaced grid values inside one interval.

    Args:
        low: The lower bound of the interval.
        high: The upper bound of the interval.
        sampling_scale: The sampling scale to use.
        num_points: The number of values to generate in the interval.

    Returns:
        A list of grid values inside the interval.
    """

    match sampling_scale:
        case 'log_uniform':
            if low <= 0 or high <= 0:
                raise ValueError(
                    'Log-uniform grid search requires positive interval bounds.'
                )
            return [
                float(value)
                for value in np.exp(
                    np.linspace(np.log(low), np.log(high), num_points)
                )
            ]
        case 'uniform':
            return [
                float(value) for value in np.linspace(low, high, num_points)
            ]
