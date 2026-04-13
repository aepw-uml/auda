from typing import cast, override

import numpy as np
from step.plot.plotter import RegressionPlotter


class TheilSenRegressionPlotter(RegressionPlotter):
    """Plots Theil-Sen regression results."""

    @override
    def plot(self) -> None:
        self.curve_description = 'Theil-Sen Trend'
        self.plot_training_data()
        self.plot_test_data()
        self.plot_curve()
        self.set_labels_legend()
        self.ax.set_title('Theil-Sen Regression')

    @override
    def plot_training_data(self) -> None:
        """Plots the training series and highlights the fitted window."""

        self.ax.plot(
            self.X_train.ravel(),
            self.y_train,
            color='lightsteelblue',
            marker='o',
            label='Training Series',
        )

        window_size = cast(int | None, self.parameters.get('window_size'))
        if window_size is None:
            raise ValueError(
                'Window size is required to plot Theil-Sen regression.'
            )

        X_window = self.X_train[-window_size:]
        y_window = self.y_train[-window_size:]
        self.ax.scatter(
            X_window.ravel(),
            y_window,
            color='cornflowerblue',
            edgecolor='black',
            label='Fitted Window',
            zorder=3,
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
        """Skips the default predicted-point scatter plot."""

        return

    @override
    def plot_curve(self) -> None:
        """Plots the fitted Theil-Sen trend from the fitted window onward."""

        window_size = cast(int | None, self.parameters.get('window_size'))
        if window_size is None:
            raise ValueError(
                'Window size is required to plot Theil-Sen regression.'
            )

        start_x = int(np.floor(self.X_train[-window_size:, 0].min()))
        _, x_max = self.get_domain()
        end_x = int(np.ceil(x_max))
        x_values = np.arange(start_x, end_x + 1, dtype=float).reshape(-1, 1)
        y_values = self.model(x_values)

        self.ax.plot(
            x_values.ravel(),
            y_values,
            color='orange',
            label=self.curve_description,
        )
