from logging import Logger
from typing import Callable, cast

import numpy as np
from common.metrics import RegressionMetricName, RegressionMetrics
from step.tuner.random_search import (
    Hyperparameters,
    HyperparameterScore,
    Interval,
    SamplingScale,
    SearchSpace,
    random_search,
)


def multistage_random_search(
    hyperparameter_names: list[str],
    search_space: SearchSpace,
    elite_fractions: list[float],
    refinement_width_rates: list[list[float]],
    evaluate_hyperparameters: Callable[[Hyperparameters], RegressionMetrics],
    sampling_scales: list[SamplingScale],
    hyperparameter_domains: list[Interval] | None = None,
    metric: RegressionMetricName = 'mape',
    expect_higher: bool | str = 'auto',
    num_iterations: list[int] = [100, 20, 10],
    seed: int = 42,
    logger: Logger | None = None,
) -> tuple[list[list[HyperparameterScore]], Hyperparameters]:
    """Runs Multistage random search (MRS) over multiple refinement stages.

    This function performs ``len(elite_fractions) + 1`` stages. At each stage
    it calls ``single_stage`` to evaluate ``num_iterations`` randomly sampled
    hyperparameter vectors within the current search space. After every
    non-final stage, the current stage's top-performing candidates are used to
    build a refined search space for the next stage. Refinement also stops
    early when the best score of the current stage does not improve enough over
    the best score of the previous stage for the chosen metric. The final
    stage evaluates the last refined space without producing a further
    refinement. The best hyperparameters are selected from the last executed
    stage only.

    Args:
        hyperparameter_names: A list of hyperparameter names.
        search_space: Initial search space for hyperparameters.
        elite_fractions: List of fractions of top candidates to consider as
            elite for each stage.
        refinement_width_rates: List of lists of domain-relative refinement
            rates for each stage.
        evaluate_hyperparameters: A function that takes a list of hyperparameter
            values and returns a RegressionMetrics object containing the
            evaluation metrics for those hyperparameters.
        sampling_scales: A list of sampling scales for each hyperparameter.
        hyperparameter_domains: Overall domains for each hyperparameter. If
            None, it will be set to the first intervals of the search space.
        metric: The name of the metric to optimize.
        expect_higher: Whether to expect higher values of the metric to be
            better. If set to 'auto', it will be determined based on the metric
            name (e.g., 'r2' is expected to be higher, while 'mape' is expected
            to be lower).
        num_iterations: Number of iterations for each stage.
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
    if len(refinement_width_rates) != num_stages:
        raise ValueError(
            'Length of refinement_width_rates must match length of '
            'elite_fractions.'
        )

    all_hyperparameter_scores: list[list[HyperparameterScore]] = []
    hyperparameter_scores: list[HyperparameterScore] = []
    previous_best_score: float | None = None
    for stage_index in range(num_stages + 1):
        if logger is not None:
            logger.info(
                f'Starting stage {stage_index + 1} of {num_stages + 1}...'
            )

        if stage_index < num_stages:
            elite_fraction = elite_fractions[stage_index]
            refinement_width_rate = refinement_width_rates[stage_index]
        else:
            elite_fraction = None
            refinement_width_rate = None

        hyperparameter_scores, search_space = single_stage(
            hyperparameter_names,
            search_space,
            elite_fraction,
            refinement_width_rate,
            evaluate_hyperparameters,
            sampling_scales,
            hyperparameter_domains,
            metric=metric,
            expect_higher=expect_higher,
            seed=seed,
            num_iterations=num_iterations[stage_index],
            logger=logger,
        )

        all_hyperparameter_scores.append(hyperparameter_scores)

        sorted_stage_hyperparameter_scores: list[HyperparameterScore] = sorted(
            hyperparameter_scores,
            key=lambda x: x[0],
            reverse=expect_higher,
        )
        current_best_score = sorted_stage_hyperparameter_scores[0][0]

        if (
            previous_best_score is not None
            and stage_index < num_stages
            and not has_meaningful_stage_improvement(
                metric,
                previous_best_score,
                current_best_score,
            )
        ):
            if logger is not None:
                logger.info(
                    'Stopping refinement early because the latest stage did '
                    'not improve enough to justify another refinement.'
                )
            break

        previous_best_score = current_best_score

    sorted_final_hyperparameter_scores: list[HyperparameterScore] = sorted(
        hyperparameter_scores, key=lambda x: x[0], reverse=expect_higher
    )
    best_hyperparameters = sorted_final_hyperparameter_scores[0][1]

    return all_hyperparameter_scores, best_hyperparameters


def has_meaningful_stage_improvement(
    metric: RegressionMetricName,
    previous_best_score: float,
    current_best_score: float,
) -> bool:
    """Determines whether the latest stage improved enough to refine again.

    The improvement thresholds are intentionally hardcoded for tiny-dataset
    tuning:
    - ``mape`` must improve by at least 1 percentage point in absolute terms
      or by at least 5 percent relative to the previous best score.
    - ``mae`` and ``rmse`` must improve by at least 5 percent relative to the
      previous best score.
    - ``r2`` must improve by at least 0.02 in absolute terms.

    Args:
        metric: The metric being optimized.
        previous_best_score: The best score from the previous stage.
        current_best_score: The best score from the current stage.

    Returns:
        Whether the current stage improved enough to justify further
        refinement.
    """

    match metric:
        case 'mape':
            absolute_improvement = previous_best_score - current_best_score
            relative_improvement = absolute_improvement / max(
                abs(previous_best_score), 1e-12
            )
            return absolute_improvement >= 0.01 or relative_improvement >= 0.05
        case 'mae' | 'rmse':
            relative_improvement = (
                previous_best_score - current_best_score
            ) / max(abs(previous_best_score), 1e-12)
            return relative_improvement >= 0.05
        case 'r2':
            absolute_gain = current_best_score - previous_best_score
            return absolute_gain >= 0.02


def single_stage(
    hyperparameter_names: list[str],
    search_space: list[list[Interval]],
    elite_fraction: float | None,
    refinement_width_rates: list[float] | None,
    evaluate_hyperparameters: Callable[[Hyperparameters], RegressionMetrics],
    sampling_scales: list[SamplingScale],
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
    ``elite_fraction`` and ``refinement_width_rates`` are provided, it keeps
    the top ``max(1, int(num_scores * elite_fraction))`` candidates as elites.
    For each elite candidate and each hyperparameter, it creates a new
    interval centered on the elite value with a half-width derived from the
    configured domain-relative refinement rate and clipped to the
    corresponding overall hyperparameter domain. These intervals become the
    search space for the next stage. If refinement parameters are ``None``,
    this function returns the current stage results and the original search
    space unchanged.

    Args:
        hyperparameter_names: A list of hyperparameter names.
        search_space: Initial search space for hyperparameters.
        elite_fraction: Fraction of top candidates to keep as elite for this
            stage.
        refinement_width_rates: List of domain-relative refinement rates for
            this stage.
        evaluate_hyperparameters: A function that takes a list of hyperparameter
            values and returns a RegressionMetrics object containing the
            evaluation metrics for those hyperparameters.
        sampling_scales: A list of sampling scales for each hyperparameter.
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
        sampling_scales,
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
    # indicated by elite_fraction or refinement_width_rates being None.
    if elite_fraction is None or refinement_width_rates is None:
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
            refinement_width_rate = refinement_width_rates[i]
            sampling_scale = sampling_scales[i]

            next_search_space[i].append(
                refine_interval(
                    hp,
                    refinement_width_rate,
                    hyperparameter_domains[i],
                    sampling_scale,
                )
            )

    return hyperparameter_scores, next_search_space


def refine_interval(
    center: float,
    refinement_width_rate: float,
    domain: Interval,
    sampling_scale: SamplingScale,
) -> Interval:
    """Builds a refined interval around a candidate hyperparameter value.

    For ``uniform`` sampling, the half-width is
    ``refinement_width_rate * (max - min)`` in the original parameter space.
    For ``log_uniform`` sampling, the half-width is
    ``refinement_width_rate * (log(max) - log(min))`` in log-space.

    Args:
        center: The elite candidate value to refine around.
        refinement_width_rate: Domain-relative half-width rate in the
            appropriate sampling space.
        domain: Overall minimum and maximum allowed values.
        sampling_scale: The sampling scale used for this hyperparameter.

    Returns:
        A refined interval clipped to the provided domain.

    Raises:
        ValueError: If log-uniform refinement is requested for a non-positive
            value or domain.
    """

    minimum, maximum = domain
    if not 0.0 < refinement_width_rate <= 1.0:
        raise ValueError(
            'refinement_width_rate must be greater than 0.0 and less than '
            'or equal to 1.0.'
        )

    match sampling_scale:
        case 'uniform':
            half_width = refinement_width_rate * (maximum - minimum)
            return (
                max(center - half_width, minimum),
                min(center + half_width, maximum),
            )
        case 'log_uniform':
            if center <= 0.0 or minimum <= 0.0 or maximum <= 0.0:
                raise ValueError(
                    'Log-uniform refinement requires positive values and '
                    'domains.'
                )

            log_center = np.log(center)
            log_minimum = np.log(minimum)
            log_maximum = np.log(maximum)
            half_width = refinement_width_rate * (log_maximum - log_minimum)

            return (
                float(np.exp(max(log_center - half_width, log_minimum))),
                float(np.exp(min(log_center + half_width, log_maximum))),
            )
