from typing import List, override

from auda.dat.models import ModelType
from auda.dat.predictors import verify_previous_model
from auda.utils.pipeline import IOSpec, Task, task

from .__common import PLOTTER_KIND, PlotterISName, PlotterOSName


@task(
    id='PL-FI',
    kind=PLOTTER_KIND,
    description='Visualizes feature importances from a trained model.',
    input_specs={
        PlotterISName.LABEL: IOSpec(dtype=str),
        PlotterISName.FEATURE_NAMES: IOSpec(dtype=List[str]),
        PlotterISName.UNITS: IOSpec(dtype=List[str]),
        PlotterISName.TITLE: IOSpec(dtype=str, required=False, default=''),
        PlotterISName.MODEL_TYPE: IOSpec(dtype=str),
        PlotterISName.FEATURE_IMPORTANCES: IOSpec(dtype=List[float]),
        PlotterISName.FEATURE_NAMES: IOSpec(dtype=List[str], required=False),
    },
    output_specs={
        PlotterOSName.FIGURE: IOSpec(dtype=object),
    },
)
class FeatureImportancesPlotter(Task):
    @override
    def run(self) -> None:
        import matplotlib.pyplot as plt
        import numpy as np
        import seaborn as sns

        verify_previous_model(self, ModelType.RANDOM_FOREST)
        feature_importances = np.array(
            self.get_input(PlotterISName.FEATURE_IMPORTANCES)
        )
        feature_names = self.get_input(PlotterISName.FEATURE_NAMES)

        if feature_names is None or len(feature_names) != len(feature_importances):
            feature_names = [f'Feature {i}' for i in range(len(feature_importances))]

        # ---- Sort features by importance (descending)
        sorted_idx = np.argsort(feature_importances)[::-1]
        feature_importances = feature_importances[sorted_idx]
        feature_names = np.array(feature_names)[sorted_idx]

        # ---- Plot
        sns.set_theme(style='whitegrid')
        fig, ax = plt.subplots(figsize=(8, 5))

        title = self.get_input(PlotterISName.TITLE)
        bars = ax.barh(feature_names, feature_importances, color='cornflowerblue')
        ax.set_xlabel('Importance', fontsize=11)
        ax.set_ylabel('Features', fontsize=11)
        ax.set_title(title, fontsize=13, pad=12)
        ax.invert_yaxis()  # highest importance on top

        # ---- Add percentage labels
        for bar in bars:
            width = bar.get_width()
            ax.text(
                width + 0.01,
                bar.get_y() + bar.get_height() / 2,
                f'{width:.2f}',
                va='center',
                ha='left',
                fontsize=9,
            )

        fig.tight_layout()

        # ---- Populate outputs
        self.set_output(PlotterISName.FIGURE, fig)
