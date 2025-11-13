from typing import List, Tuple, override

import numpy as np

from auda.dat.datasets import (
    LabeledSamples,
    get_feature_label,
    verify_feature_space_dimension,
)
from auda.dat.transformers import standardize_x
from auda.utils.pipeline import IOSpec, Task, task

from .__common import PLOTTER_KIND, PlotterISName, PlotterOSName, extend_range


@task(
    id='PL-PR',
    kind=PLOTTER_KIND,
    description='Plots polynomial regression curves with sample data.',
    input_specs={
        PlotterISName.SAMPLES: IOSpec(dtype=LabeledSamples),
        PlotterISName.LABEL: IOSpec(dtype=str),
        PlotterISName.FEATURE_NAMES: IOSpec(dtype=List[str]),
        PlotterISName.UNITS: IOSpec(dtype=List[str]),
        PlotterISName.TITLE: IOSpec(dtype=str, required=False, default=''),
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
        label = self.get_input(PlotterISName.LABEL)
        title = self.get_input(PlotterISName.TITLE)
        ax.set_xlabel(feature_name)
        ax.set_ylabel(label)
        ax.set_title(title)

        ax.legend()
        fig.tight_layout()

        # ---- Populate outputs
        self.set_output(PlotterOSName.FIGURE, fig)


@task(
    id='PL-PR-2D',
    kind=PLOTTER_KIND,
    description='Plots a 2D Polynomial Regression surface (trained on standardized X) '
    'with sample points.',
    input_specs={
        PlotterISName.SAMPLES: IOSpec(dtype=LabeledSamples),
        PlotterISName.LABEL: IOSpec(dtype=str),
        PlotterISName.FEATURE_NAMES: IOSpec(dtype=List[str]),
        PlotterISName.UNITS: IOSpec(dtype=List[str]),
        PlotterISName.TITLE: IOSpec(dtype=str, required=False),
        PlotterISName.DEGREE: IOSpec(dtype=int),
        PlotterISName.INTERCEPT: IOSpec(dtype=float),
        PlotterISName.COEFFICIENTS: IOSpec(dtype=List[float]),
        PlotterISName.COEFFICIENTS_EXPONENTS: IOSpec(dtype=List[Tuple[int]]),
        PlotterISName.X_MEAN: IOSpec(dtype=List[float]),
        PlotterISName.X_STANDARD_DEVIATION: IOSpec(dtype=List[float]),
    },
    output_specs={
        PlotterOSName.FIGURE: IOSpec(dtype=object),
    },
)
class PolynomialRegression2DPlotter(Task):
    @override
    def run(self) -> None:
        """
        Visualize z = Σ θ_{i,j} * x^i * y^j (fit on standardized X) as a 3D surface,
        with the original samples overlaid as a scatter. The evaluation is done by:
          1) building a grid in ORIGINAL feature space,
          2) standardizing the grid with provided X mean/std,
          3) evaluating the polynomial using (intercept, coefficients, exponents).
        """
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

        # ---- Inputs
        samples: LabeledSamples = self.get_input(PlotterISName.SAMPLES)
        verify_feature_space_dimension(samples, expected_dimension=2)

        feature_names: List[str] = self.get_input(PlotterISName.FEATURE_NAMES) or [
            'x1',
            'x2',
        ]
        units: List[str] = self.get_input(PlotterISName.UNITS)
        label: str = self.get_input(PlotterISName.LABEL) or 'Target'
        title: str = self.get_input(PlotterISName.TITLE) or ''

        degree: int = int(self.get_input(PlotterISName.DEGREE))
        intercept: float = float(self.get_input(PlotterISName.INTERCEPT))
        coeffs: List[float] = list(self.get_input(PlotterISName.COEFFICIENTS))
        exps: List[Tuple[int, int]] = self.get_input(
            PlotterISName.COEFFICIENTS_EXPONENTS
        )

        # Ensure (0,0) term is explicit and aligned with intercept
        if not exps or tuple(exps[0]) == (0, 0):
            exps = [e for e in exps if tuple(e) != (0, 0)]
        full_exps: List[Tuple[int, int]] = [(0, 0)] + exps
        theta = np.array([intercept] + coeffs, dtype=float)

        # ---- Extract original data (for scatter)
        X = np.array([x for x, _ in samples], dtype=float)  # shape (n, 2)
        z_true = np.array([z for _, z in samples], dtype=float)

        # ---- Build plot domain in original units with a small pad
        x1_min, x1_max = float(X[:, 0].min()), float(X[:, 0].max())
        x2_min, x2_max = float(X[:, 1].min()), float(X[:, 1].max())
        x1_min, x1_max = extend_range(x1_min, x1_max, margin_ratio=0.05)
        x2_min, x2_max = extend_range(x2_min, x2_max, margin_ratio=0.05)

        # Reasonable resolution (increase if you need smoother surfaces)
        n1, n2 = 60, 60
        x1_lin = np.linspace(x1_min, x1_max, n1)
        x2_lin = np.linspace(x2_min, x2_max, n2)
        X1, X2 = np.meshgrid(x1_lin, x2_lin)  # shapes (n2, n1)

        # ---- Standardize grid using provided stats
        grid = np.column_stack([X1.ravel(), X2.ravel()])  # (n1*n2, 2)
        grid_std = standardize_x(self, grid)  # standardized features

        xs = grid_std[:, 0]
        ys = grid_std[:, 1]

        # ---- Evaluate polynomial on standardized grid
        # z = Σ_k theta[k] * (xs^i_k) * (ys^j_k)
        Z = np.zeros_like(xs, dtype=float)
        for k, (i, j) in enumerate(full_exps):
            if i == 0 and j == 0:
                Z += theta[k]
            else:
                Z += theta[k] * (xs**i) * (ys**j)
        Z = Z.reshape(X1.shape)

        # ---- Plot
        plt.close('all')
        fig = plt.figure(figsize=(9, 6.75), dpi=150)
        ax = fig.add_subplot(111, projection='3d')

        # Surface
        surf = ax.plot_surface(
            X1,
            X2,
            Z,
            rstride=1,
            cstride=1,
            linewidth=0,
            antialiased=True,
            alpha=0.75,
            cmap='viridis',
            zorder=1,
        )

        # Scatter of samples
        ax.scatter(
            X[:, 0],
            X[:, 1],
            z_true,  # type: ignore
            s=36,
            c=z_true,
            cmap='viridis',
            edgecolor='white',
            linewidth=0.4,
            alpha=0.95,
            zorder=3,
        )

        # Axes labels
        x_label = get_feature_label(
            feature_names[0], units[0] if len(units) > 0 else None
        )
        y_label = get_feature_label(
            feature_names[1], units[1] if len(units) > 1 else None
        )
        ax.set_xlabel(x_label, labelpad=8)
        ax.set_ylabel(y_label, labelpad=8)
        ax.set_zlabel(label, labelpad=10)
        ax.set_title(f'{title} (deg {degree})', pad=14)

        # Colorbar tied to surface
        cbar = fig.colorbar(surf, ax=ax, fraction=0.04, pad=0.06, shrink=0.85)
        cbar.set_label(label, rotation=270, labelpad=14, va='center')

        # ---- Layout polish
        fig.subplots_adjust(left=0.03, right=0.96, top=0.92, bottom=0.06)
        ax.grid(True, alpha=0.25)

        # ---- Populate outputs
        self.set_output(PlotterOSName.FIGURE, fig)
