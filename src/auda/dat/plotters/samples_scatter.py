from typing import List, override

from auda.dat.datasets import (
    DatasetOSName,
    LabeledSamples,
    verify_feature_space_dimension,
)
from auda.dat.datasets.__common import get_feature_label
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


@task(
    id='PL-SS-3D',
    kind=PLOTTER_KIND,
    description='Plots a 3D scatter of samples for datasets with two input features '
    'and one target.',
    input_specs={
        PlotterISName.SAMPLES: IOSpec(dtype=LabeledSamples),
        PlotterISName.LABEL: IOSpec(dtype=str),
        PlotterISName.FEATURE_NAMES: IOSpec(dtype=List[str]),
        PlotterISName.UNITS: IOSpec(dtype=List[str]),
        PlotterISName.TITLE: IOSpec(dtype=str, required=False, default=''),
    },
    output_specs={
        PlotterOSName.FIGURE: IOSpec(dtype=object),
    },
)
class SampleScatterPlotter3D(Task):
    @override
    def run(self) -> None:
        import matplotlib.pyplot as plt
        import numpy as np
        from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

        samples: LabeledSamples = self.get_input(PlotterISName.SAMPLES)
        verify_feature_space_dimension(samples, expected_dimension=2)

        feature_names: List[str] = self.get_input(PlotterISName.FEATURE_NAMES)
        units: List[str] = self.get_input(PlotterISName.UNITS)
        label: str = self.get_input(PlotterISName.LABEL)
        title: str = self.get_input(PlotterISName.TITLE)

        if len(feature_names) != 2:
            raise ValueError(
                'Dataset must have exactly two features for 3D scatter plot.'
            )

        # Extract coordinates
        X = np.array([x for x, _ in samples])
        y = np.array([z for _, z in samples])

        x_vals, y_vals, z_vals = X[:, 0], X[:, 1], y

        # ---- Create the 3D scatter plot
        plt.close('all')
        fig = plt.figure(figsize=(8, 6), dpi=150)
        ax = fig.add_subplot(111, projection='3d')

        sc = ax.scatter(
            x_vals,
            y_vals,
            z_vals,  # type: ignore
            c=z_vals,
            cmap='viridis',
            alpha=0.8,
            s=40,
            edgecolor='white',
            linewidth=0.3,
        )

        # ---- Axis labels
        x_label = get_feature_label(feature_names[0], units[0])
        y_label = get_feature_label(feature_names[1], units[1])
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        ax.set_zlabel(label, labelpad=5)
        ax.set_title(title, pad=18)
        ax.zaxis.offsetText

        # ---- Improve layout and colorbar placement
        fig.subplots_adjust(left=0.05, right=0.84, top=0.92, bottom=0.08)
        cbar = fig.colorbar(
            sc,
            ax=ax,
            fraction=0.045,  # slightly wider space reserved
            pad=0.12,  # more padding between plot and bar
            shrink=0.9,  # full height
        )
        cbar.set_label(label, rotation=270, labelpad=20, va='center')

        ax.grid(True, alpha=0.3)

        # Let tight_layout recompute margins after manual subplots_adjust
        fig.tight_layout(rect=[0, 0, 0.88, 1])  # type: ignore

        self.set_output(PlotterOSName.FIGURE, fig)
