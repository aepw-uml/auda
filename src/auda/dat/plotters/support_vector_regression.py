from typing import List, override

from auda.dat.datasets import LabeledSamples
from auda.dat.plotters import PlotterISName
from auda.dat.transformers import reverse_x, reverse_y, standardize_x
from auda.utils.pipeline import IOSpec, Task, task

from .__common import PLOTTER_KIND, PlotterOSName, extend_range


@task(
    id='PL-SVR',
    kind=PLOTTER_KIND,
    description='Generates predictions using a trained Support Vector Regression '
    '(SVR) model.',
    input_specs={
        PlotterISName.LABEL: IOSpec(dtype=str),
        PlotterISName.FEATURE_NAMES: IOSpec(dtype=List[str]),
        PlotterISName.UNITS: IOSpec(dtype=List[str]),
        PlotterISName.TITLE: IOSpec(dtype=str, required=False),
        PlotterISName.MODEL_TYPE: IOSpec(dtype=str),
        PlotterISName.MODEL: IOSpec(dtype=object),
        PlotterISName.SAMPLES: IOSpec(dtype=LabeledSamples),
        PlotterISName.EPSILON: IOSpec(dtype=float),
        PlotterISName.SUPPORT_INDICES: IOSpec(dtype=List[int]),
        PlotterISName.X_MEAN: IOSpec(dtype=float),
        PlotterISName.X_STANDARD_DEVIATION: IOSpec(dtype=float),
        PlotterISName.Y_MEAN: IOSpec(dtype=float),
        PlotterISName.Y_STANDARD_DEVIATION: IOSpec(dtype=float),
        PlotterISName.PREDICTION_SAMPLES: IOSpec(dtype=LabeledSamples, required=False),
    },
    output_specs={
        PlotterOSName.FIGURE: IOSpec(dtype=object),
    },
)
class SupportVectorRegressionPlotter(Task):
    @override
    def run(self) -> None:
        from itertools import chain

        import matplotlib.pyplot as plt
        import numpy as np

        from auda.dat.models import create_curve

        samples = self.get_input(PlotterISName.SAMPLES)

        # Create a figure and axis
        fig, ax = plt.subplots()

        # ---- Scatter Original Samples ----
        x_true = [x for x, _ in samples]
        y_true = [y for _, y in samples]

        ax.scatter(
            x_true,
            y_true,
            s=60,
            linewidths=0,
            zorder=5,
            label='Original Samples',
            edgecolors='white',
            color='cornflowerblue',
        )

        # ---- Scatter Support Vectors ----
        support_indices = self.get_input(PlotterISName.SUPPORT_INDICES)
        x_support = [x_true[i] for i in support_indices]
        y_support = [y_true[i] for i in support_indices]

        ax.scatter(
            x_support,
            y_support,
            s=60,
            linewidths=0.75,
            color='cornflowerblue',
            edgecolors='black',
            zorder=5,
            label='Support Vectors',
        )

        # ---- Scatter Prediction Samples (if exists) ----
        prediction_samples = self.get_input(PlotterISName.PREDICTION_SAMPLES)
        x_pred = None
        if prediction_samples:
            x_pred = [x for x, _ in prediction_samples]
            y_pred = [y for _, y in prediction_samples]

            ax.scatter(
                x_pred,
                y_pred,
                s=60,
                linewidths=0.75,
                zorder=5,
                label='Prediction Samples',
                color='orange',
                edgecolors='black',
            )

        # ---- Prepare SVR Curve ----
        domain = list(chain.from_iterable(x_true + (x_pred or [])))
        x_min, x_max = min(domain), max(domain)

        # Standardize
        x_range = np.array([x_min, x_max])
        x_range_std = standardize_x(self, x_range).tolist()
        x_min_std, x_max_std = x_range_std[0], x_range_std[1]
        x_min_std, x_max_std = extend_range(x_min_std, x_max_std)

        svr_model = self.get_input(PlotterISName.MODEL)
        curve_resolution = 300
        x_curve_std = np.linspace(x_min_std, x_max_std, curve_resolution)
        y_curve_std = svr_model.predict(x_curve_std.reshape(-1, 1))

        # Reverse standardization for plotting
        x_curve = reverse_x(self, x_curve_std)
        y_curve = reverse_y(self, y_curve_std)

        curve = create_curve(x_curve.tolist(), y_curve.tolist())

        # ---- Draw SVR Fitted Curve
        # Epsilon tube in original units
        epsilon = self.get_input(PlotterISName.EPSILON)
        y_std = self.get_input(PlotterISName.Y_STANDARD_DEVIATION)

        # Epsilon tube in original units
        epsilon_original = None
        if (
            epsilon is not None
            and y_std is not None
            and np.isfinite(epsilon)
            and np.isfinite(y_std)
        ):
            epsilon_original = epsilon * y_std
            if not np.isfinite(epsilon_original) or epsilon_original <= 0:
                epsilon_original = None

        # Draw the curve and epsilon tube
        if curve and epsilon_original:
            x_curve_std, y_curve_std = zip(*curve)
            x_curve_std = np.array(x_curve_std, dtype=float).flatten()
            y_curve_std = np.array(y_curve_std, dtype=float)
            ax.fill_between(
                x_curve_std,
                y_curve_std - epsilon_original,
                y_curve_std + epsilon_original,
                alpha=0.25,
                label='ε-tube',
                zorder=2,
                color='orange',
            )

            ax.plot(
                x_curve_std,
                y_curve_std,
                lw=2.5,
                color='orange',
                label='SVR Prediction' if x_pred else 'SVR Fit',
                zorder=3,
            )

        # ---- Highlight Forecast Region
        if prediction_samples and x_pred:
            last_train_year = float(max(list(chain.from_iterable(x_true))))
            last_pred_year = float(max(list(chain.from_iterable(x_pred))))
            plt.axvspan(
                last_train_year,
                last_pred_year,
                color='gray',
                alpha=0.15,
                label='Forecast Region',
            )

        # ---- Setup Plot Labels and Show
        feature_names = self.get_input(PlotterISName.FEATURE_NAMES)
        feature_name = feature_names[0] if feature_names else 'Feature 1'
        label = self.get_input(PlotterISName.LABEL)
        title = self.get_input(PlotterISName.TITLE)

        ax.set_xlabel(feature_name)
        ax.set_ylabel(label)
        ax.set_title(title)
        ax.grid(True, alpha=0.25)
        ax.legend()
        fig.tight_layout()

        # ---- Populate outputs
        self.set_output(PlotterOSName.FIGURE, fig)
