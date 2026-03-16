from typing import override

import numpy as np

from step.plot.plotter import RegressionPlotter


class ARIMARegressionPlotter(RegressionPlotter):
    """Plots ARIMA regression results."""

    @override
    def plot(self) -> None:
        self.curve_description = 'ARIMA Forecast'
        self.plot_training_data()
        self.plot_test_data()
        self.plot_curve()
        self.set_labels_legend()
        self.ax.set_title('ARIMA Regression')

    @override
    def plot_training_data(self) -> None:
        """Plots the historical training series as a line chart."""

        self.ax.plot(
            self.X_train.ravel(),
            self.y_train,
            color='cornflowerblue',
            marker='o',
            label='Training Series',
        )

    @override
    def plot_test_data(self) -> None:
        """Plots the held-out test series as a line chart."""

        if self.X_test is None or self.y_test is None:
            return

        self.ax.plot(
            self.X_test.ravel(),
            self.y_test,
            color='green',
            marker='o',
            label='Test Series',
        )

    @override
    def plot_pred_data(self) -> None:
        """Skips the default predicted-point scatter plot for ARIMA."""

        return

    @override
    def plot_curve(self) -> None:
        """Plots ARIMA forecasts as a forecast line over integer years."""

        last_train_x = int(np.max(self.X_train[:, 0]))
        last_train_y = float(self.y_train[-1])
        _, x_max = self.get_domain()
        max_future_x = int(np.floor(x_max))

        if max_future_x <= last_train_x:
            return

        future_x = np.arange(last_train_x + 1, max_future_x + 1).reshape(-1, 1)
        future_y = self.model(future_x)
        x_values = np.concatenate(
            (
                np.array([[last_train_x]], dtype=float),
                future_x.astype(float),
            ),
            axis=0,
        )
        y_values = np.concatenate(([last_train_y], future_y), axis=0)

        self.ax.plot(
            x_values.ravel(),
            y_values,
            color='orange',
            marker='o',
            label=self.curve_description,
        )
