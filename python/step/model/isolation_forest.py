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
    seed: int = 471,
    include_target: bool = True,
) -> IsolationForestResult:
    """Fits an isolation forest model and filters detected outliers.

    Args:
        X: Training feature matrix.
        y: Training target vector.
        contamination: Expected proportion of outliers in the training set.
        seed: Random seed used for reproducibility.
        include_target: Whether to include the target variable when fitting and
            applying the isolation forest.

    Returns:
        An ``IsolationForestResult`` containing the fitted model and the
        filtered training data.
    """

    data = X
    if include_target:
        data = np.column_stack([X, y.reshape(-1, 1)])

    model = IsolationForest(
        contamination=contamination,  # type: ignore
        n_estimators=200,
        random_state=seed,
    )
    model.fit(data)

    inlier_mask = model.predict(data) == 1

    return IsolationForestResult(
        model=model,
        inlier_mask=inlier_mask,
        X_inliers=X[inlier_mask],
        y_inliers=y[inlier_mask],
    )
