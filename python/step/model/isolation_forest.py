from dataclasses import dataclass

import numpy as np
from sklearn.ensemble import IsolationForest


@dataclass(frozen=True)
class IsolationForestResult:
    """Stores the fitted model and the filtered training data.

    Attributes:
        model: Fitted isolation forest model.
        inlier_mask: Boolean mask indicating which rows are kept.
        X_inliers: Feature matrix after removing detected outliers.
        y_inliers: Target vector after removing detected outliers.
    """

    model: IsolationForest
    inlier_mask: np.ndarray
    X_inliers: np.ndarray
    y_inliers: np.ndarray


def isolation_forest(
    X: np.ndarray,
    y: np.ndarray,
    contamination: float = 0.05,
    seed: int = 417,
) -> IsolationForest:
    """Fits an isolation forest model on the feature matrix.

    Args:
        X: Training feature matrix.
        y: Training target vector. This argument is accepted to match the
            project step interface, but the model is fit on ``X`` only.
        contamination: Expected proportion of outliers in the training set.
        seed: Random seed used for reproducibility.

    Returns:
        A fitted ``IsolationForest`` model.
    """

    _ = y
    model = IsolationForest(
        contamination=contamination,  # type: ignore
        n_estimators=200,
        random_state=seed,
    )
    model.fit(X)

    return model


def remove_outliers(
    X: np.ndarray,
    y: np.ndarray,
    contamination: float = 0.05,
    seed: int = 417,
) -> IsolationForestResult:
    """Fits isolation forest and removes rows predicted as outliers.

    Args:
        X: Training feature matrix.
        y: Training target vector.
        contamination: Expected proportion of outliers in the training set.
        seed: Random seed used for reproducibility.

    Returns:
        The fitted model, inlier mask, and filtered training data.
    """

    model = isolation_forest(
        X=X,
        y=y,
        contamination=contamination,
        seed=seed,
    )
    inlier_mask = model.fit_predict(X) == 1

    return IsolationForestResult(
        model=model,
        inlier_mask=inlier_mask,
        X_inliers=X[inlier_mask],
        y_inliers=y[inlier_mask],
    )
