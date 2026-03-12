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
    """Performs time series cross-validation.

    Args:
        X: The input features as a 2D numpy array of shape (m, n).
        y: The target values as a 1D numpy array of shape (m,).
        evaluate_fold: A function that takes training features, training
            targets, validation features, and validation targets, and returns a
            RegressionMetrics object containing the evaluation metrics for that
            fold.
        num_k_folds: The number of folds to use for cross-validation. Must be
            greater than 1 and less than or equal to m // 2.
    """

    m = X.shape[0]

    # Check num_k_folds.
    if num_k_folds <= 1 or num_k_folds > (m // 2):
        raise ValueError(
            f'num_k_folds must be > 1 and <= {m // 2}, but got {num_k_folds}.'
        )

    # Evaluation metrics for each fold.
    all_metrics: list[RegressionMetrics] = []

    folds = np.array_split(np.arange(m), num_k_folds)
    for fold_idx in range(1, num_k_folds):
        X_train = np.concatenate(
            [X[folds[j]] for j in range(num_k_folds) if j < fold_idx]
        )
        y_train = np.concatenate(
            [y[folds[j]] for j in range(num_k_folds) if j < fold_idx]
        )
        val_idx = folds[fold_idx]
        X_val = X[val_idx]
        y_val = y[val_idx]

        regressionMetrics = evaluate_fold(X_train, y_train, X_val, y_val)
        all_metrics.append(regressionMetrics)

    return all_metrics
