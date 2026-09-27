"""Constructs annual training arrays without compressing missing years."""

import numpy as np


def annual_training_series(
    X: np.ndarray, y: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Reindexes observed training values onto a complete annual grid.

    Args:
        X: Calendar years with shape ``(n_samples, 1)``.
        y: Finite observed targets with shape ``(n_samples,)``.

    Returns:
        Annual years and corresponding targets, with NaN at missing years.
        The grid ends at the last observed training year.

    Raises:
        ValueError: If years are nonintegral, duplicated, or inputs invalid.
    """

    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    if (
        X.ndim != 2 or X.shape[1] != 1 or y.ndim != 1
        or len(X) != len(y) or len(y) == 0
        or not np.isfinite(X).all() or not np.isfinite(y).all()
    ):
        raise ValueError('Expected finite, nonempty annual observations.')
    times = X[:, 0]
    if not np.allclose(times, np.round(times), rtol=0, atol=1e-8):
        raise ValueError('Annual models require integer calendar years.')
    times = np.round(times).astype(int)
    if len(np.unique(times)) != len(times):
        raise ValueError('Annual models require unique calendar years.')
    years = np.arange(times.min(), times.max() + 1)
    values = np.full(len(years), np.nan)
    values[times - years[0]] = y
    return years, values
