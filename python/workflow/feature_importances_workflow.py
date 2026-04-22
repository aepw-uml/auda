from pathlib import Path
from typing import override

from common.dataset import Dataset, DatasetSchema
from common.files import save_content_to_file
from common.workflow.workflow import Workflow
from step.model.isolation_forest import isolation_forest
from step.model.random_forest_regression import RandomForestRegression
from step.plot.feature_importances import FeatureImportancesPlotter


class FeatureImportancesflow(Workflow):
    @override
    def run(self, dataset: Dataset, schema: DatasetSchema, **context) -> None:
        super().run(**context)

        X, y = dataset.X, dataset.y
        if y is None:
            raise ValueError(
                'Target variable "y" is required for feature importances '
                'calculation.'
            )

        # Remove outliers using isolation forest
        contamination = float(context.get('contamination', 0.05))
        seed = int(context.get('seed', 417))
        result = isolation_forest(X, y, contamination=contamination, seed=seed)
        X = result.X_inliers
        y = result.y_inliers

        # Calculate importances
        regression = RandomForestRegression({}, **context).fit(X, y)
        feature_importances = regression.parameters['feature_importances']
        sorted_indices = feature_importances.argsort()[::-1]

        report_str = 'Feature importances: '
        report_str += ', '.join(
            f'{schema.feature_names[index]} ({feature_importances[index]:.4f})'
            for index in sorted_indices
        )
        print(report_str)

        dir_path = Path('results') / 'feature_importances'
        save_content_to_file(dir_path / 'report.txt', report_str)

        # Plot importances and save it to a file
        plotter = FeatureImportancesPlotter(schema, '')
        plotter.plot(feature_importances)

        file_path = plotter.save(dir_path / 'feature_importances')
        print(f'Saved feature importances plot to "{file_path}".')
