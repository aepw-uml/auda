from typing import List, override

from auda.dat.datasets import DatasetOSName, LabeledSamples
from auda.utils.pipeline import IOSpec, Task, task

from .__common import PLOTTER_KIND, PlotterISName, PlotterOSName


@task(
    id='PL-SS',
    kind=PLOTTER_KIND,
    description='Plots scatter points of sample data.',
    input_specs={
        PlotterISName.SAMPLES: IOSpec(dtype=LabeledSamples),
        PlotterISName.LABEL: IOSpec(dtype=str),
        PlotterISName.FEATURE_NAMES: IOSpec(dtype=List[str]),
        PlotterISName.UNITS: IOSpec(dtype=List[str]),
        PlotterISName.TITLE: IOSpec(dtype=str, required=False),
    },
    output_specs={
        PlotterOSName.FIGURE: IOSpec(dtype=object),
    },
)
class SampleScatterPlotter(Task):
    @override
    def run(self) -> None:
        import matplotlib.pyplot as plt
        import seaborn as sns

        feature_names: List[str] = self.get_input(PlotterISName.FEATURE_NAMES)
        if len(feature_names) > 1:
            raise ValueError('Dataset must have exactly one feature for scatter plot')

        samples: LabeledSamples = self.get_input(DatasetOSName.SAMPLES)
        x_values = [x[0] for x, _ in samples]
        y_values = [y for _, y in samples]

        # ---- Create scatter plot
        sns.set_theme(style='whitegrid')
        fig, ax = plt.subplots()
        sns.scatterplot(x=x_values, y=y_values, ax=ax)

        # Label axes based on metadata
        feature_name = feature_names[0] if feature_names else 'Feature'
        label_name = self.get_input('label')

        ax.set_xlabel(feature_name)
        ax.set_ylabel(label_name)
        ax.set_title(self.get_input('title') or '')

        # ---- Populate outputs
        self.set_output(PlotterISName.FIGURE, fig)
