from logging import Logger
from random import Random
from typing import Callable

from common.metrics import RegressionMetricName, RegressionMetrics

Interval = tuple[float, float]
Hyperparameters = list[float]
HyperparameterScore = tuple[float, Hyperparameters]
Domain = list[Interval]
SearchSpace = list[Domain]


def random_search(
    hyperparameter_names: list[str],
    search_space: SearchSpace,
    evaluate_hyperparameters: Callable[[Hyperparameters], RegressionMetrics],
    metric: RegressionMetricName = 'mape',
    num_iterations: int = 100,
    seed: int = 42,
    logger: Logger | None = None,
) -> list[HyperparameterScore]:
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
            values and returns a RegressionMetrics object containing the
            evaluation metrics for those hyperparameters.
        metric: The name of the metric to optimize.
        num_iterations: The number of random hyperparameter combinations to
            evaluate.
        seed: The random seed for reproducibility.
        logger: An optional logger to log the progress of the random search.

    Returns:
        A list of ``(score, hyperparameters)`` tuples in the order they were
        evaluated.
    """

    rng = Random(seed)
    if len(search_space) != len(hyperparameter_names):
        raise ValueError(
            'Length of search_space must match length of hyperparameter_names.'
        )

    hyperparameter_scores: list[HyperparameterScore] = []
    for i in range(num_iterations):
        hyperparameters: Hyperparameters = [
            select_random_hyperparameter(intervals, rng)
            for intervals in search_space
        ]

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
                f'[{i + 1}/{num_iterations}] '
                f'Hyperparameters = ({hyperparameters_str}); '
                f'score = {score:.3e}'
            )

    return hyperparameter_scores


def select_random_hyperparameter(intervals: Domain, rng: Random) -> float:
    """Selects a random hyperparameter value from the given intervals.

    Args:
        intervals: A list of intervals from which to select a random value.
        rng: A random number generator instance.

    Returns:
        A random hyperparameter value selected from the given intervals.
    """

    selected_interval_index = rng.randint(0, len(intervals) - 1)
    selected_interval = intervals[selected_interval_index]

    return rng.uniform(selected_interval[0], selected_interval[1])
