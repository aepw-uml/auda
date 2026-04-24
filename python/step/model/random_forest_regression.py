from typing import Self, override

import numpy as np
from sklearn.ensemble import RandomForestRegressor
from step.model.model import SupervisedLearningModel


class RandomForestRegression(SupervisedLearningModel):
    @override
    def fit(
        self, X: np.ndarray, y: np.ndarray, num_features: int | None = None
    ) -> Self:
        super().fit(X, y, num_features)

        random_forest_regressor = RandomForestRegressor(
            n_estimators=300,
            max_depth=None,
            min_samples_split=2,
            min_samples_leaf=1,
            random_state=471,
            n_jobs=-1,
        )
        random_forest_regressor.fit(X, y)

        self.parameters['feature_importances'] = (
            random_forest_regressor.feature_importances_
        )

        return self
