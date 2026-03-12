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
    expect_higher: bool | str = 'auto',
    seed: int = 42,
    num_iterations: int = 100,
    logger: Logger | None = None,
) -> list[HyperparameterScore]:
    """Perform random search to find the best hyperparameters.

    Args:
        hyperparameter_names: A list of hyperparameter names.
        search_space: A list of lists of intervals, where each inner list
            corresponds to a hyperparameter and contains the intervals from
            which to select random values.
        evaluate_hyperparameters: A function that takes a list of hyperparameter
            values and returns a RegressionMetrics object containing the
            evaluation metrics for those hyperparameters.
        metric: The name of the metric to optimize.
        expect_higher: Whether to expect higher values of the metric to be
            better. If set to 'auto', it will be determined based on the metric
            name (e.g., 'r2' is expected to be higher, while 'mape' is expected
            to be lower).
        seed: The random seed for reproducibility.
        num_iterations: The number of random hyperparameter combinations to
            evaluate.
        logger: An optional logger to log the progress of the random search.

    Returns:
        A list of tuples, where each tuple contains a score and the
        corresponding hyperparameters that achieved that score.
    """

    if expect_higher == 'auto':
        expect_higher = metric == 'r2'

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
        score: float = metrics.get_value(metric)
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
                f'Hyperparameters = {hyperparameters_str}, '
                f'score = {score}'
            )

    return hyperparameter_scores


def select_random_hyperparameter(intervals: Domain, rng: Random) -> float:
    """Select a random hyperparameter value from the given intervals.

    Args:
        intervals: A list of intervals from which to select a random value.
        rng: A random number generator instance.

    Returns:
        A random hyperparameter value selected from the given intervals.
    """

    selected_interval_index = rng.randint(0, len(intervals) - 1)
    selected_interval = intervals[selected_interval_index]

    return rng.uniform(selected_interval[0], selected_interval[1])
