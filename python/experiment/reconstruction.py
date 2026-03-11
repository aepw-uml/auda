from typing import Type, override

import numpy as np
from experiment.experiment import RegressionExperiment
from sklearn.base import RegressorMixin
from sklearn.ensemble import IsolationForest
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


class Reconstruction(RegressionExperiment):
    @override
    def __init__(
        self,
        name: str,
        description: str,
        seed: int = 42,
        train_rate: float = 0.8,
        val_rate: float = 0.0,
        regressor: Type[RegressorMixin] = LinearRegression,
    ) -> None:
        super().__init__(name, description, seed, train_rate, val_rate)
        self.regressor = regressor

    @override
    def train(self) -> None:
        super().train()

        X_train, y_train = self.get_training_set()
        use_isolation_forest: bool = self.context.get(
            'use_isolation_forest', False
        )
        if use_isolation_forest:
            contamination = self.context.get('contamination', 'auto')
            iso = IsolationForest(
                contamination=contamination,
                random_state=self.seed,
            )
            inlier_mask = iso.fit_predict(self.X_train) == 1
            self.context['inlier_mask'] = inlier_mask

            X_train_inliers: np.ndarray = X_train[inlier_mask]
            y_train_inliers: np.ndarray = y_train[inlier_mask]

        self.pipeline = Pipeline(
            [
                ('scaler', StandardScaler()),
                (
                    self.regressor.__name__,
                    self.regressor(**self.context),
                ),
            ]
        )

        if use_isolation_forest:
            self.pipeline.fit(X_train_inliers, y_train_inliers)
        else:
            self.pipeline.fit(X_train, self.y_train)
