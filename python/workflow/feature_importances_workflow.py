from typing import override

from common.dataset import Dataset, DatasetSchema
from common.workflow.workflow import Workflow
from step.model.random_forest_regression import RandomForestRegression


class FeatureImportancesflow(Workflow):
    @override
    def run(self, dataset: Dataset, schema: DatasetSchema, **context) -> None:
        # Calculate importances
        X, y = dataset.X, dataset.y
        if y is None:
            raise ValueError(
                'Target variable "y" is required for feature importances '
                'calculation.'
            )
        regression = RandomForestRegression({}, **context).fit(X, y)
        feature_importances = regression.parameters['feature_importances']

        print(feature_importances)
