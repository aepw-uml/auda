import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.utils.validation import check_array, check_is_fitted, check_X_y


class NaivePersistence(RegressorMixin, BaseEstimator):
    def fit(self, X: np.ndarray, y: np.ndarray):
        X, y = check_X_y(X, y)

        if X.shape[1] != 1:
            raise ValueError('NaivePersistenceModel expects a single feature.')

        order = np.argsort(X[:, 0])
        self.last_y_ = float(y[order][-1])

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        check_is_fitted(self, 'last_y_')
        X = check_array(X)

        if X.shape[1] != 1:
            raise ValueError('NaivePersistenceModel expects a single feature.')

        m = X.shape[0]

        return np.full((m,), self.last_y_, dtype=float)
