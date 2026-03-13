from typing import override

from step.plot.plotter import RegressionPlotter


class PolynomialRegression(RegressionPlotter):
    @override
    def plot(self) -> None:
        super().plot()
        self.ax.set_title('Polynomial Regression')
