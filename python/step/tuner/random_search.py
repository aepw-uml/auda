from logging import Logger
from typing import Callable

import numpy as np
from common.metrics import RegressionMetrics
from step.tuner.types import (
    Configuration,
    Domain,
    Hyperparameters,
    SamplingScale,
    SearchSpace,
)


def random_search(
    hyperparameter_names: list[str],
    search_space: SearchSpace,
    evaluate_hyperparameters: Callable[
        [Hyperparameters], list[RegressionMetrics]
    ],
    sampling_scales: list[SamplingScale],
    num_iterations: int = 100,
    seed: int = 42,
    logger: Logger | None = None,
) -> list[Configuration]:
    """Samples and evaluates random hyperparameter combinations.

    For each iteration, this function draws one value for each hyperparameter
    by first choosing one interval uniformly at random from that
    hyperparameter's domain and then sampling uniformly within the chosen
    interval. The resulting hyperparameter vector is passed to
    ``evaluate_hyperparameters``, and the requested metric value is stored
    together with the sampled hyperparameters. The function returns all sampled
    `(score, hyperparameters)` pairs in evaluation order and does not sort or
    rank them.

    Args:
        hyperparameter_names: A list of hyperparameter names.
        search_space: A list of lists of intervals, where each inner list
            corresponds to a hyperparameter and contains the intervals from
            which to select random values.
        evaluate_hyperparameters: A function that takes a list of hyperparameter
            values and returns the per-fold regression metrics for those
            hyperparameters.
        sampling_scales: A list of sampling scales for each hyperparameter.
        metric: The name of the metric to optimize.
        num_iterations: The number of random hyperparameter combinations to
            evaluate.
        seed: The random seed for reproducibility.
        logger: An optional logger to log the progress of the random search.

    Returns:
        A list of ``(hyperparameters, metrics_list)`` tuples in the order they
        were evaluated.
    """

    rng = np.random.default_rng(seed=seed)
    if len(search_space) != len(hyperparameter_names):
        raise ValueError(
            'Length of search_space must match length of hyperparameter_names.'
        )

    hyperparameter_scores: list[Configuration] = []
    for i in range(num_iterations):
        hyperparameters: Hyperparameters = [
            select_random_hyperparameter(intervals, rng, sampling_scales[i])
            for i, intervals in enumerate(search_space)
        ]

        metrics_list: list[RegressionMetrics] = evaluate_hyperparameters(
            hyperparameters
        )
        hyperparameter_scores.append((hyperparameters, metrics_list))

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
                f'[{i + 1}/{num_iterations}] '
                f'Hyperparameters = ({hyperparameters_str}).'
            )

    return hyperparameter_scores


def select_random_hyperparameter(
    intervals: Domain,
    rng: np.random.Generator,
    sampling_scale: SamplingScale,
) -> float:
    """Selects a random hyperparameter value from the given intervals.

    This function first randomly selects one of the given intervals and then
    samples a value from that interval according to the specified sampling scale
    (either uniform or log-uniform).

    Args:
        intervals: A list of intervals from which to select a random value.
        rng: A random number generator instance.
        sampling_scale: The sampling scale to use.

    Returns:
        A random hyperparameter value selected from the given intervals.
    """

    # Randomly select one of the intervals.
    selected_interval_index = rng.integers(0, len(intervals))
    low, high = intervals[selected_interval_index]

    # Randomly select a value from the selected interval.
    match sampling_scale:
        case 'log_uniform':
            return np.exp(rng.uniform(np.log(low), np.log(high)))
        case 'uniform':
            return rng.uniform(low, high)
