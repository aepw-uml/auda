from logging import Logger
from typing import Callable

from common.metrics import RegressionMetricName, RegressionMetrics
from step.tuner import random_search
from step.tuner.random_search import (
    Hyperparameters,
    HyperparameterScore,
    Interval,
    SearchSpace,
    random_search,
)


def multistage_random_search(
    hyperparameter_names: list[str],
    search_space: SearchSpace,
    elite_fractions: list[float],
    refinement_widths: list[list[float]],
    evaluate_hyperparameters: Callable[[Hyperparameters], RegressionMetrics],
    hyperparameter_domains: list[Interval] | None = None,
    metric: RegressionMetricName = 'mape',
    expect_higher: bool | str = 'auto',
    seed: int = 42,
    num_iterations: int = 100,
    logger: Logger | None = None,
) -> list[list[HyperparameterScore]]:
    """Performs multistage random search tuning.

    Args:
        hyperparameter_names: A list of hyperparameter names.
        search_space: Initial search space for hyperparameters.
        elite_fractions: List of fractions of top candidates to consider as
            elite for each stage.
        refinement_widths: List of lists of widths for refining the search space
            around elite candidates for each stage.
        evaluate_hyperparameters: A function that takes a list of hyperparameter
            values and returns a RegressionMetrics object containing the
            evaluation metrics for those hyperparameters.
        hyperparameter_domains: Overall domains for each hyperparameter. If
            None, it will be set to the first intervals of the search space.
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
        A list of lists of hyperparameter scores for each stage.
    """

    if hyperparameter_domains is None:
        hyperparameter_domains = [intervals[0] for intervals in search_space]

    num_stages: int = len(elite_fractions)
    if len(refinement_widths) != num_stages:
        raise ValueError(
            'Length of refinement_widths must match length of elite_fractions.'
        )

    all_hyperparameter_scores: list[list[HyperparameterScore]] = []
    hyperparameter_scores: list[HyperparameterScore] = []
    for stage_index in range(num_stages + 1):
        if logger is not None:
            logger.info(
                f'Starting stage {stage_index + 1} of {num_stages + 1}...'
            )

        hyperparameter_scores, search_space = single_stage(
            hyperparameter_names,
            search_space,
            elite_fractions[stage_index],
            refinement_widths[stage_index],
            evaluate_hyperparameters,
            hyperparameter_domains,
            metric=metric,
            expect_higher=expect_higher,
            seed=seed,
            num_iterations=num_iterations,
            logger=logger,
        )

        all_hyperparameter_scores.append(hyperparameter_scores)

    return all_hyperparameter_scores


def single_stage(
    hyperparameter_names: list[str],
    search_space: list[list[Interval]],
    elite_fraction: float,
    refinement_widths: list[float],
    evaluate_hyperparameters: Callable[[Hyperparameters], RegressionMetrics],
    hyperparameter_domains: list[Interval],
    metric: RegressionMetricName = 'mape',
    expect_higher: bool | str = 'auto',
    seed: int = 42,
    num_iterations: int = 100,
    logger: Logger | None = None,
) -> tuple[list[HyperparameterScore], SearchSpace]:
    """Performs a single stage of random search tuning.

    Args:
        hyperparameter_names: A list of hyperparameter names.
        search_space: Initial search space for hyperparameters.
        elite_fractions: List of fractions of top candidates to consider as
            elite for each stage.
        refinement_widths: List of lists of widths for refining the search space
            around elite candidates for each stage.
        evaluate_hyperparameters: A function that takes a list of hyperparameter
            values and returns a RegressionMetrics object containing the
            evaluation metrics for those hyperparameters.
        hyperparameter_domains: Overall domains for each hyperparameter.
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
        A tuple containing a list of hyperparameter scores and the next search
        space for the next stage.
    """

    hyperparameter_scores: list[HyperparameterScore] = random_search(
        hyperparameter_names,
        search_space,
        evaluate_hyperparameters,
        metric=metric,
        expect_higher=expect_higher,
        seed=seed,
        num_iterations=num_iterations,
        logger=logger,
    )
    num_scores: int = len(hyperparameter_scores)
    sorted_hyperparameter_scores: list[HyperparameterScore] = sorted(
        hyperparameter_scores, key=lambda x: x[0], reverse=True
    )

    num_elite_candidates = max(1, int(num_scores * elite_fraction))
    elite_candidates: list[Hyperparameters] = [
        hyperparameters
        for _, hyperparameters in sorted_hyperparameter_scores[
            :num_elite_candidates
        ]
    ]

    num_hyperparameters: int = len(hyperparameter_names)
    next_search_space: SearchSpace = [[] for _ in range(num_hyperparameters)]
    for candidate in elite_candidates:
        for i in range(len(candidate)):
            hp = candidate[i]
            refinement_width = refinement_widths[i]
            minimum = hyperparameter_domains[i][0]
            maximum = hyperparameter_domains[i][1]
            next_search_space[i].append(
                (
                    max(hp - refinement_width, minimum),
                    min(hp + refinement_width, maximum),
                )
            )

    return hyperparameter_scores, next_search_space
