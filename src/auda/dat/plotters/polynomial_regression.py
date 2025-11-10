from typing import List, override

import numpy as np

from auda.dat.datasets import LabeledSamples
from auda.utils.pipeline import IOSpec, Task, task

from .__common import PLOTTER_KIND, PlotterISName, PlotterOSName


@task(
    id='PL-PR',
    kind=PLOTTER_KIND,
    description='Plots polynomial regression curves with sample data.',
    input_specs={
        PlotterISName.SAMPLES: IOSpec(dtype=LabeledSamples),
        PlotterISName.LABEL: IOSpec(dtype=str),
        PlotterISName.FEATURE_NAMES: IOSpec(dtype=List[str]),
        PlotterISName.UNITS: IOSpec(dtype=List[str]),
        PlotterISName.TITLE: IOSpec(dtype=str, required=False),
        PlotterISName.INTERCEPT: IOSpec(dtype=float),
        PlotterISName.COEFFICIENTS: IOSpec(dtype=List[float]),
        PlotterISName.X_MEAN: IOSpec(dtype=List[float]),
        PlotterISName.X_STANDARD_DEVIATION: IOSpec(dtype=List[float]),
    },
    output_specs={
        PlotterOSName.FIGURE: IOSpec(dtype=object),
    },
)
class PolynomialRegressionPlotter(Task):
    @override
    def run(self) -> None:
        """
        Plots a centered polynomial regression curve for a single-feature labeled
        dataset. The polynomial is evaluated as:

            y = polyval(x - x_mean, [intercept] + coefficients)
        """
        import matplotlib.pyplot as plt
        import seaborn as sns

        intercept = self.get_input(PlotterISName.INTERCEPT)
        coefficients = self.get_input(PlotterISName.COEFFICIENTS)
        x_mean_list = self.get_input(PlotterISName.X_MEAN)
        x_std_list = self.get_input(PlotterISName.X_STANDARD_DEVIATION)

        x_mean = float(x_mean_list[0])
        x_std = float(x_std_list[0])
        coeffs = np.array([intercept] + coefficients, dtype=float)

        samples = self.get_input(PlotterISName.SAMPLES)
        X = np.array([x for x, _ in samples])
        y = np.array([y for _, y in samples])
        x_vals = X[:, 0]

        # ---- Create a smooth x-grid with a bit of padding
        x_min, x_max = float(x_vals.min()), float(x_vals.max())
        pad = (x_max - x_min) * 0.03 if x_max > x_min else 1.0
        x_plot = np.linspace(x_min - pad, x_max + pad, 400)

        # ---- Evaluate polynomial in the space the model was trained on: z = (x - μ)/σ
        z_plot = (x_plot - x_mean) / x_std
        y_plot = np.polynomial.polynomial.polyval(z_plot, coeffs)

        # ---- Plot
        sns.set_theme(style='whitegrid')
        fig, ax = plt.subplots()

        sns.scatterplot(
            x=x_vals,
            y=y,
            ax=ax,
            s=60,
            linewidth=0.7,
            edgecolor='white',
            label='Samples',
        )

        # Fitted curve
        deg = len(coeffs) - 1
        ax.plot(
            x_plot,
            y_plot,
            linewidth=2,
            label=f'Polynomial fit (deg {deg})',
            color='orange',
        )

        # Labels & title
        feature_names = self.get_input(PlotterISName.FEATURE_NAMES)
        feature_name = feature_names[0] if feature_names else 'Feature'
        label = self.get_input(PlotterISName.LABEL) or ''
        title = self.get_input(PlotterISName.TITLE) or ''
        ax.set_xlabel(feature_name)
        ax.set_ylabel(label)
        ax.set_title(title)

        ax.legend()
        fig.tight_layout()

        # ---- Populate outputs
        self.set_output(PlotterOSName.FIGURE, fig)
