from typing import List, override

from auda.dat.datasets import LabeledSamples
from auda.utils.pipeline import IOSpec, Task, task

from .__common import (
    MODEL_KIND,
    ModelISName,
    ModelOSName,
    ModelType,
    split_labeled_samples,
)


@task(
    id='MD-RF',
    kind=MODEL_KIND,
    description='Trains a Random Forest regressor for feature importance and '
    'prediction.',
    input_specs={
        ModelISName.SAMPLES: IOSpec(dtype=LabeledSamples),
    },
    output_specs={
        ModelOSName.MODEL_TYPE: IOSpec(dtype=str),
        ModelOSName.MODEL: IOSpec(dtype=object),
        ModelOSName.FEATURE_IMPORTANCES: IOSpec(dtype=List[float]),
    },
)
class RandomForestRegressorModel(Task):
    """
    Train a Random Forest regressor on labeled samples and expose the fitted model.
    """

    @override
    def run(self) -> None:
        from sklearn.ensemble import RandomForestRegressor

        # ---- Get data
        samples = self.get_input(ModelISName.SAMPLES)
        x, y = split_labeled_samples(samples)

        # ---- Train a random forest regressor
        random_forest_regressor = RandomForestRegressor(
            n_estimators=300,
            max_depth=None,
            min_samples_split=2,
            min_samples_leaf=1,
            random_state=42,
            n_jobs=-1,
        )
        random_forest_regressor.fit(x, y)

        # ---- Results
        feature_importances = random_forest_regressor.feature_importances_

        # ---- Populate outputs
        self.set_output(ModelOSName.MODEL_TYPE, ModelType.RANDOM_FOREST)
        self.set_output(ModelOSName.MODEL, random_forest_regressor)
        self.set_output(ModelOSName.FEATURE_IMPORTANCES, feature_importances)
