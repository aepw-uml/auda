from statistics import mean
from typing import Callable

from step.tuner.types import Hyperparameters


def one_standard_error(
    trials: list[tuple[Hyperparameters, list[float]]],
    complexity_key: Callable[[Hyperparameters], tuple[float, ...]],
    *,
    prefer_lower=True,
) -> Hyperparameters:
    """
    Selects the hyperparameter setting that is within one standard error of the
    best trial and has the lowest complexity.

    Args:
        trials: A list of tuples, where each tuple contains the evaluated
            hyperparameters and their per-fold scores.
        complexity_key: A function that returns a sorting key where lower
            values correspond to simpler hyperparameter settings.
        prefer_lower: Whether to prefer lower scores over higher scores. If
            True, trials with mean scores less than or equal to the threshold
            will be selected. If False, trials with mean
            scores greater than or equal to the threshold will be selected.

    Returns:
        The hyperparameter setting that is within one standard error of the
        best trial and has the lowest complexity.

    Raises:
        ValueError: If no trials are provided.
    """

    if len(trials) == 0:
        raise ValueError('No trials provided.')

    candidates = one_standard_error_candidates(
        trials, prefer_lower=prefer_lower
    )
    return min(candidates, key=complexity_key)


def one_standard_error_candidates(
    trials: list[tuple[Hyperparameters, list[float]]],
    *,
    prefer_lower=True,
) -> list[Hyperparameters]:
    """Selects trial candidates that are within one standard error of the best
    trial.

    Args:
        trials: A list of tuples, where each tuple contains the evaluated
            hyperparameters and their per-fold scores.
        prefer_lower: Whether to prefer lower scores over higher scores. If
            True, trials with mean scores less than or equal to the threshold
            will be selected. If False, trials with mean
            scores greater than or equal to the threshold will be selected.

    Returns:
        Hyperparameter settings belonging to trials within one standard error
        of the best trial.
    """

    if len(trials) == 0:
        raise ValueError('No trials provided.')

    trial_means: list[tuple[Hyperparameters, float]] = [
        (hyperparameters, mean(scores)) for hyperparameters, scores in trials
    ]

    best_trial_mean = trial_means[0][1]
    best_trial_scores = trials[0][1]
    for i in range(1, len(trial_means)):
        _, trial_mean = trial_means[i]
        if trial_mean < best_trial_mean:
            best_trial_mean = trial_mean
            best_trial_scores = trials[i][1]

    threshold = best_trial_mean + 0.1 * standard_error(best_trial_scores)

    def is_within_threshold(mean_score: float) -> bool:
        return (
            mean_score <= threshold if prefer_lower else mean_score >= threshold
        )

    return [
        hyperparameters
        for hyperparameters, mean_score in trial_means
        if is_within_threshold(mean_score)
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
