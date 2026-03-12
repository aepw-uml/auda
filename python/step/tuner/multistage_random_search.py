from logging import Logger
from typing import Callable, cast

from common.metrics import RegressionMetricName, RegressionMetrics
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
    num_iterations: int = 50,
    seed: int = 42,
    logger: Logger | None = None,
) -> tuple[list[list[HyperparameterScore]], Hyperparameters]:
    """Runs random search over multiple refinement stages.

    This function performs ``len(elite_fractions) + 1`` stages. At each stage
    it calls ``single_stage`` to evaluate ``num_iterations`` randomly sampled
    hyperparameter vectors within the current search space. After every
    non-final stage, the current stage's top-performing candidates are used to
    build a refined search space for the next stage. The final stage evaluates
    the last refined space without producing a further refinement. The best
    hyperparameters are selected from the final stage only.

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
        num_iterations: The number of random hyperparameter combinations to
            evaluate.
        seed: The random seed for reproducibility.
        logger: An optional logger to log the progress of the random search.

    Returns:
        A tuple containing the per-stage random-search results and the best
        hyperparameter vector from the final stage.
    """

    if hyperparameter_domains is None:
        hyperparameter_domains = [intervals[0] for intervals in search_space]

    if expect_higher == 'auto':
        expect_higher = metric == 'r2'

    expect_higher = cast(bool, expect_higher)

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

        if stage_index < num_stages:
            elite_fraction = elite_fractions[stage_index]
            refinement_width = refinement_widths[stage_index]
        else:
            elite_fraction = None
            refinement_width = None

        hyperparameter_scores, search_space = single_stage(
            hyperparameter_names,
            search_space,
            elite_fraction,
            refinement_width,
            evaluate_hyperparameters,
            hyperparameter_domains,
            metric=metric,
            expect_higher=expect_higher,
            seed=seed,
            num_iterations=num_iterations,
            logger=logger,
        )

        all_hyperparameter_scores.append(hyperparameter_scores)

    sorted_final_hyperparameter_scores: list[HyperparameterScore] = sorted(
        hyperparameter_scores, key=lambda x: x[0], reverse=expect_higher
    )
    best_hyperparameters = sorted_final_hyperparameter_scores[0][1]

    return all_hyperparameter_scores, best_hyperparameters


def single_stage(
    hyperparameter_names: list[str],
    search_space: list[list[Interval]],
    elite_fraction: float | None,
    refinement_widths: list[float] | None,
    evaluate_hyperparameters: Callable[[Hyperparameters], RegressionMetrics],
    hyperparameter_domains: list[Interval],
    metric: RegressionMetricName,
    expect_higher: bool,
    num_iterations: int,
    seed: int,
    logger: Logger | None,
) -> tuple[list[HyperparameterScore], SearchSpace]:
    """Evaluates one random-search stage and prepares the next search space.

    This function first calls ``random_search`` on the provided search space.
    It then sorts the sampled candidates by the requested metric. If
    ``elite_fraction`` and ``refinement_widths`` are provided, it keeps the top
    ``max(1, int(num_scores * elite_fraction))`` candidates as elites. For each
    elite candidate and each hyperparameter, it creates a new interval centered
    on the elite value with the configured refinement width, clipped to the
    corresponding overall hyperparameter domain. These intervals become the
    search space for the next stage. If refinement parameters are ``None``,
    this function returns the current stage results and the original search
    space unchanged.

    Args:
        hyperparameter_names: A list of hyperparameter names.
        search_space: Initial search space for hyperparameters.
        elite_fraction: Fraction of top candidates to keep as elite for this
            stage.
        refinement_widths: List of lists of widths for refining the search space
            around elite candidates for this stage.
        evaluate_hyperparameters: A function that takes a list of hyperparameter
            values and returns a RegressionMetrics object containing the
            evaluation metrics for those hyperparameters.
        hyperparameter_domains: Overall domains for each hyperparameter.
        metric: The name of the metric to optimize.
        expect_higher: Whether higher values of the chosen metric are better.
        num_iterations: The number of random hyperparameter combinations to
            evaluate.
        seed: The random seed for reproducibility.
        logger: An optional logger to log the progress of the random search.

    Returns:
        A tuple containing the sampled ``(score, hyperparameters)`` pairs for
        this stage and the search space to use for the next stage.
    """

    hyperparameter_scores: list[HyperparameterScore] = random_search(
        hyperparameter_names,
        search_space,
        evaluate_hyperparameters,
        metric=metric,
        seed=seed,
        num_iterations=num_iterations,
        logger=logger,
    )
    num_scores: int = len(hyperparameter_scores)
    sorted_hyperparameter_scores: list[HyperparameterScore] = sorted(
        hyperparameter_scores, key=lambda x: x[0], reverse=expect_higher
    )

    # Stop computing the next search space if we are in the last stage,
    # indicated by elite_fraction or refinement_widths being None.
    if elite_fraction is None or refinement_widths is None:
        return hyperparameter_scores, search_space

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
