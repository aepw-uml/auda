import numpy as np
from auda.step.dataset import DatasetSchema
from auda.step.plot import PlotStep
from auda.step.plot.plot_set import PlotSet
from auda.step.spec import Spec
from auda.utils.pipeline import IOValueMap, Pipeline, step


@step(
    id='PL-FI',
    description='Visualizes feature importances from a trained model.',
    input_specs=[
        Spec.FEATURE_IMPORTANCES,
        Spec.DATASET_SCHEMA,
    ],
    output_specs=[Spec.FIGURE, Spec.AXES],
)
class PlotFeatureImportances(PlotStep):
    def run(
        self,
        feature_importances: np.ndarray,
        dataset_schema: DatasetSchema,
    ) -> IOValueMap:
        import matplotlib.pyplot as plt

        n_features = len(feature_importances)
        fig_height = max(6, 1.5 * n_features)
        figure, axes = plt.subplots(figsize=(14, fig_height))

        feature_names = dataset_schema.feature_names

        # ---- Sort features by importance (descending)
        sorted_idx = np.argsort(feature_importances)[::-1]
        feature_importances = feature_importances[sorted_idx]
        feature_names = np.array(feature_names)[sorted_idx]

        # ---- Plot horizontal bar chart
        bars = axes.barh(
            feature_names,
            feature_importances,
            height=0.60,
            color='cornflowerblue',
        )
        axes.set_xlabel('Importance', fontsize=13)
        axes.set_ylabel('Features', fontsize=13)
        axes.invert_yaxis()

        # ---- Add percentage labels
        axes.bar_label(
            bars,
            labels=[f'{v:.2f}' for v in feature_importances],
            padding=3,
            fontsize=13,
        )

        Pipeline([PlotSet]).run(
            {
                Spec.FIGURE.name: figure,
                Spec.AXES.name: axes,
            }
        )

        return self.regular_output(figure, axes)
