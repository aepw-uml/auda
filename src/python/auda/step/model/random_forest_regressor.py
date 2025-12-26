from auda.step.dataset import DatasetBasedStep
from auda.step.spec import Dataset, Spec
from auda.utils.pipeline import IOValueMap, step


@step(
    id='MD-RF',
    description='Trains a Random Forest regressor for feature importance and '
    'prediction.',
    input_specs=[Spec.ON.optional(Spec.DATASET.name)],
    output_specs=[Spec.MODEL, Spec.FEATURE_IMPORTANCES],
)
class RandomForestRegressorModel(DatasetBasedStep):
    def run(self, on: str | Dataset) -> IOValueMap:
        from sklearn.ensemble import RandomForestRegressor

        X, y = self.get_dataset_from_on(on)

        # ---- Train a random forest regressor
        model = RandomForestRegressor(
            n_estimators=300,
            max_depth=None,
            min_samples_split=2,
            min_samples_leaf=1,
            random_state=42,
            n_jobs=-1,
        )
        model.fit(X, y)

        return {
            Spec.MODEL.name: model,
            Spec.FEATURE_IMPORTANCES.name: model.feature_importances_,
        }
