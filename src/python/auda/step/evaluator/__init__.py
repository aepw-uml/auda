import numpy as np


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Calculates Mean Absolute Error between true and predicted values.

    Args:
        y_true: True values.
        y_pred: Predicted values.

    Returns:
        Mean Absolute Error.
    """

    return float(np.mean(np.abs(y_true - y_pred)))


def mse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Calculates Mean Squared Error between true and predicted values.

    Args:
        y_true: True values.
        y_pred: Predicted values.

    Returns:
        Mean Squared Error.
    """

    return float(np.mean((y_true - y_pred) ** 2))


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Calculates Root Mean Squared Error between true and predicted values.

    Args:
        y_true: True values.
        y_pred: Predicted values.

    Returns:
        Root Mean Squared Error.
    """

    return np.sqrt(mse(y_true, y_pred))


def r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Calculates R-squared (Coefficient of Determination) between true and
    predicted values.

    Args:
        y_true: True values.
        y_pred: Predicted values.

    Returns:
        R-squared value.
    """

    ss_total = np.sum((y_true - np.mean(y_true)) ** 2)
    ss_residual = np.sum((y_true - y_pred) ** 2)

    if ss_total == 0:
        return 0.0

    return 1 - (ss_residual / ss_total)


def mape(
    y_true: np.ndarray, y_pred: np.ndarray, epsilon: float = 1e-8
) -> float:
    """Calculates Mean Absolute Percentage Error between true and predicted
    values.

    Args:
        y_true: True values.
        y_pred: Predicted values.
        epsilon: Small value to avoid division by zero.

    Returns:
        Mean Absolute Percentage Error.
    """

    denom = np.where(np.abs(y_true) < epsilon, np.nan, np.abs(y_true))

    return float(np.nanmean(np.abs((y_true - y_pred) / denom) * 100.0))
