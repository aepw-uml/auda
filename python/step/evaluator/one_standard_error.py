from statistics import mean
from typing import Callable

from step.tuner.types import Hyperparameters


def one_standard_error(
    configurations: list[tuple[Hyperparameters, list[float]]],
    complexity_key: Callable[[Hyperparameters], tuple[float, ...]],
    *,
    prefer_lower=True,
) -> Hyperparameters:
    """
    Selects the hyperparameter configuration that is within one standard error
    of the best configuration and has the lowest complexity.

    Args:
        configurations: A list of tuples, where each tuple contains a
            hyperparameter configuration and a list of scores for that
            configuration.
        complexity_key: A function that returns a sorting key where lower
            values correspond to simpler hyperparameter configurations.
        prefer_lower: Whether to prefer lower scores over higher scores. If
            True, configurations with mean scores less than or equal to the
            threshold will be selected. If False, configurations with mean
            scores greater than or equal to the threshold will be selected.

    Returns:
        The hyperparameter configuration that is within one standard error of
        the best configuration and has the lowest complexity.

    Raises:
        ValueError: If no configurations are provided.
    """

    if len(configurations) == 0:
        raise ValueError('No configurations provided.')

    candidates = one_standard_error_candidates(
        configurations, prefer_lower=prefer_lower
    )
    return min(candidates, key=complexity_key)


def one_standard_error_candidates(
    configurations: list[tuple[Hyperparameters, list[float]]],
    *,
    prefer_lower=True,
) -> list[Hyperparameters]:
    """Selects hyperparameter configurations that are within one standard error
    of the best configuration.

    Args:
        configurations: A list of tuples, where each tuple contains a
            hyperparameter configuration and a list of scores for that
            configuration.
        prefer_lower: Whether to prefer lower scores over higher scores. If
            True, configurations with mean scores less than or equal to the
            threshold will be selected. If False, configurations with mean
            scores greater than or equal to the threshold will be selected.

    Returns:
        A list of hyperparameter configurations that are within one standard
        error of the best configuration.
    """

    if len(configurations) == 0:
        raise ValueError('No configurations provided.')

    configuration_means: list[tuple[Hyperparameters, float]] = [
        (configuration, mean(scores))
        for configuration, scores in configurations
    ]

    # Find the mean and scores of the configuration with the lowest mean score.
    best_configuration_mean = configuration_means[0][1]
    best_configuration_scores = configurations[0][1]
    for i in range(1, len(configuration_means)):
        _, configuration_mean = configuration_means[i]
        if configuration_mean < best_configuration_mean:
            best_configuration_mean = configuration_mean
            best_configuration_scores = configurations[i][1]

    # Calculate the threshold as the mean of the best configuration plus one
    # standard error.
    threshold = best_configuration_mean + 0.1 * standard_error(
        best_configuration_scores
    )

    # Only keep configurations whose mean score is less than or equal to the
    # threshold.
    return [
        configuration
        for configuration, mean in configuration_means
        if prefer_lower
        and mean <= threshold
        or not prefer_lower
        and mean >= threshold
    ]


def standard_error(scores: list[float]) -> float:
    """Calculates the standard error of the mean for a list of scores.

    The standard error of the mean is calculated as the standard deviation of
    the scores divided by the square root of the number of scores.

    Args:
        scores: A list of scores.

    Returns:
        The standard error of the mean for the scores.
    """

    n = len(scores)
    if n <= 1:
        return 0.0

    mean = sum(scores) / n
    variance = sum((x - mean) ** 2 for x in scores) / (n - 1)
    return (variance / n) ** 0.5
