from typing import Callable

import numpy as np
from common.metrics import RegressionMetrics


def time_series_cross_validation(
    X: np.ndarray,
    y: np.ndarray,
    evaluate_fold: Callable[
        [np.ndarray, np.ndarray, np.ndarray, np.ndarray], RegressionMetrics
    ],
    num_k_folds: int = 4,
) -> list[RegressionMetrics]:
    """Performs expanding-window time series cross-validation.

    This function preserves chronological order. It first splits the sample
    indices into ``num_k_folds + 1`` contiguous blocks with
    ``np.array_split(np.arange(m), num_k_folds + 1)``. The first block is used
    only as the initial training window. For validation fold ``i``, the
    training set is the concatenation of all blocks before ``i``, and the
    validation set is block ``i`` itself. This produces ``num_k_folds``
    validation runs, each evaluated on data that occurs strictly after its
    training data.

    Args:
        X: The input features as a 2D numpy array of shape (m, n).
        y: The target values as a 1D numpy array of shape (m,).
        evaluate_fold: A function that takes training features, training
            targets, validation features, and validation targets, and returns a
            RegressionMetrics object containing the evaluation metrics for that
            fold.
        num_k_folds: The number of validation folds to evaluate. The function
            creates ``num_k_folds + 1`` chronological blocks so that the first
            block can serve as the initial training window.

    Returns:
        A list of ``RegressionMetrics`` objects, one for each validation fold,
        ordered from earliest to latest validation window.
    """

    m = X.shape[0]

    if X.ndim != 2:
        raise ValueError(f'X must be 2D, got shape {X.shape}.')

    if y.ndim != 1:
        raise ValueError(f'y must be 1D, got shape {y.shape}.')

    if len(y) != m:
        raise ValueError(
            f'X and y must have the same number of rows, got {m} and {len(y)}.'
        )

    if num_k_folds <= 0 or num_k_folds >= m:
        raise ValueError(
            f'num_k_folds must be > 0 and < {m}, but got {num_k_folds}.'
        )

    all_metrics: list[RegressionMetrics] = []

    # Create a list of indices for each fold. If num_k_folds is 4 and m is 10,
    # then folds will be [[0, 1], [2, 3], [4, 5], [6, 7], [8, 9]]. One extra
    # block is needed so the first block can serve as initial training data.
    folds: list[np.ndarray] = np.array_split(np.arange(m), num_k_folds + 1)

    for fold_idx in range(1, num_k_folds + 1):
        train_idx = np.concatenate(folds[:fold_idx])
        val_idx = folds[fold_idx]

        X_train = X[train_idx]
        y_train = y[train_idx]
        X_val = X[val_idx]
        y_val = y[val_idx]

        metrics = evaluate_fold(X_train, y_train, X_val, y_val)
        all_metrics.append(metrics)

    return all_metrics
