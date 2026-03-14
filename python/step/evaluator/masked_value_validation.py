from typing import Callable

import numpy as np
from common.metrics import RegressionMetrics


def masked_value_validation(
    X: np.ndarray,
    y: np.ndarray,
    evaluate: Callable[
        [np.ndarray, np.ndarray, np.ndarray, np.ndarray], RegressionMetrics
    ],
    validation_rate: float = 0.2,
    seed: int = 42,
) -> RegressionMetrics:
    """Evaluates a model on a randomly masked validation subset.

    This function samples a random subset of rows for validation and uses the
    remaining rows for training. It is intended for situations where
    chronological structure does not need to be preserved.

    Args:
        X: The input features as a 2D numpy array of shape ``(m, n)``.
        y: The target values as a 1D numpy array of shape ``(m,)``.
        evaluate: A function that takes training features, training targets,
            validation features, and validation targets, and returns a
            ``RegressionMetrics`` object.
        validation_rate: Fraction of rows to reserve for validation.
        seed: Random seed used when sampling the validation rows.

    Returns:
        The regression metrics produced by ``evaluate`` on the sampled split.

    Raises:
        ValueError: If the inputs have invalid shapes or if
            ``validation_rate`` would produce an empty training or validation
            split.
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

    if not 0.0 < validation_rate < 1.0:
        raise ValueError(
            'validation_rate must be greater than 0.0 and less than 1.0.'
        )

    num_validation = max(int(m * validation_rate), 2)
    if num_validation <= 0 or num_validation >= m:
        raise ValueError(
            'validation_rate must produce at least one validation sample and '
            'leave at least one training sample.'
        )

    rng = np.random.default_rng(seed)
    validation_idx = np.sort(rng.choice(m, size=num_validation, replace=False))
    training_mask = np.ones(m, dtype=bool)
    training_mask[validation_idx] = False

    X_train = X[training_mask]
    y_train = y[training_mask]
    X_val = X[validation_idx]
    y_val = y[validation_idx]

    return evaluate(X_train, y_train, X_val, y_val)
