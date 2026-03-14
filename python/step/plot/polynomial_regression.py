from typing import override

from step.plot.plotter import RegressionPlotter


class PolynomialRegressionPlotter(RegressionPlotter):
    @override
    def plot(self) -> None:
        super().plot()
        self.curve_description = 'Polynomial Regression'
        self.ax.set_title('Polynomial Regression')
