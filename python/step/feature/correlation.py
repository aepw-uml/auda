import numpy as np


def calculate_correlation_matrix(X: np.ndarray) -> np.ndarray:
    """Calculates the correlation matrix for the given data matrix X.

    Args:
        X: A 2D array where rows represent samples and columns represent
            features.

    Returns:
        A 2D array representing the correlation matrix of the features in X.
    """

    n = X.shape[0]
    X = X - np.mean(X, axis=0)
    covariance_matrix: np.ndarray = (X.T @ X) / (n - 1)

    std_dev = np.sqrt(np.diag(covariance_matrix))
    D_inv = np.diag(1 / std_dev)
    correlation_matrix = D_inv @ covariance_matrix @ D_inv

    return correlation_matrix
