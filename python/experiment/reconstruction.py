from typing import Type, override

from experiment.experiment import RegressionExperiment
from sklearn.base import RegressorMixin
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


class Reconstruction(RegressionExperiment):
    @override
    def __init__(
        self,
        name: str,
        description: str,
        regressor: Type[RegressorMixin],
        seed: int = 42,
        train_rate: float = 0.8,
        val_rate: float = 0.0,
    ) -> None:
        super().__init__(name, description, seed, train_rate, val_rate)
        self.regressor = regressor

    @override
    def train(self) -> None:
        super().train()

        self.pipeline = Pipeline(
            [
                ('scaler', StandardScaler()),
                # ('isolation_forest', IsolationForest()),
                (self.regressor.__name__, self.regressor()),
            ]
        )

        self.pipeline.fit(self.X_train, self.y_train)
