from typing import override

from step.plot.plotter import RegressionPlotter


class GaussianProcessRegressionPlotter(RegressionPlotter):
    """Plots Gaussian process regression results."""

    @override
    def plot(self) -> None:
        super().plot()
        self.curve_description = 'GPR Fit'
        self.ax.set_title('Gaussian Process Regression')
