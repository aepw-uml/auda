from typing import override

from step.plot.plotter import RegressionPlotter


class RidgeRegressionPlotter(RegressionPlotter):
    @override
    def plot(self) -> None:
        super().plot()
        self.curve_description = 'Ridge Regression'
        self.ax.set_title('Ridge Regression')
