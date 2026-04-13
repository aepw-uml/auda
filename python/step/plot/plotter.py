import os
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Callable, override

import matplotlib.pyplot as plt
import numpy as np
from common.dataset import DatasetSchema, combine_names_and_units
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.ticker import MaxNLocator, StrMethodFormatter


class Plotter(ABC):
    """Provides the base interface for plotter implementations."""

    def __init__(self, schema: DatasetSchema, title: str) -> None:
        """Initializes the shared plotting state for a dataset schema.

        Args:
            schema: Schema metadata used to label and configure plots.
        """

        self.schema: DatasetSchema = schema
        self.title: str = title

        # Initialize the figure and axes for plotting.
        fig, ax = plt.subplots()
        self.fig: Figure = fig
        self.ax: Axes = ax

    @abstractmethod
    def plot(self) -> None:
        """Renders the plot contents onto the configured axes."""

        pass

    def show(self) -> None:
        """Displays the current figure after adjusting the layout."""

        self.fig.tight_layout()
        plt.show()

    def save(self, file_path: Path) -> str:
        """Saves the current figure to disk and return the output path.

        Args:
            file_path: Base path used when writing the generated figure.

        Returns:
            The saved file path as a string.
        """

        self.fig.tight_layout()

        os.makedirs(os.path.dirname(file_path) or '.', exist_ok=True)
        self.fig.savefig(
            file_path,
            dpi=600,
            bbox_inches='tight',
            pad_inches=0.1,
        )

        return str(file_path) + '.png'


class RegressionPlotter(Plotter, ABC):
    """Plots training data, test data, and regression model predictions."""

    @override
    def __init__(
        self,
        schema: DatasetSchema,
        title: str,
        model: Callable[[np.ndarray], np.ndarray],
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test: np.ndarray | None,
        y_test: np.ndarray | None,
        X_pred: np.ndarray | None,
        curve_description: str = 'Model Prediction',
        parameters: dict[str, str] = {},
        hyperparameters: dict[str, str] = {},
    ) -> None:
        """Initializes the regression plotter with model inputs and outputs.

        Args:
            schema: Schema metadata used to label the chart.
            model: Callable used to predict y-values from x-values.
            X_train: Training feature values.
            y_train: Training target values.
            X_test: Test feature values.
            y_test: Test target values.
            curve_description: Optional label for the model prediction curve.
        """

        super().__init__(schema, title)

        self.model: Callable[[np.ndarray], np.ndarray] = model
        self.X_train: np.ndarray = X_train
        self.y_train: np.ndarray = y_train
        self.X_test: np.ndarray | None = X_test
        self.y_test: np.ndarray | None = y_test
        self.X_pred: np.ndarray | None = X_pred
        self.curve_description: str = curve_description
        self.parameters: dict[str, str] = parameters
        self.hyperparameters: dict[str, str] = hyperparameters

        self.y_pred: np.ndarray | None = None
        if self.X_pred is not None:
            self.y_pred = self.model(self.X_pred)

    @override
    def plot(self) -> None:
        self.plot_training_data()
        self.plot_test_data()
        self.plot_pred_data()
        self.plot_curve()
        self.set_labels_legend()

    def plot_training_data(self) -> None:
        """Plots the training samples on the current axes."""

        self.ax.scatter(
            self.X_train,
            self.y_train,
            color='cornflowerblue',
            label='Training Samples',
        )

    def plot_test_data(self) -> None:
        """Plots the test samples on the current axes."""

        if self.X_test is None or self.y_test is None:
            return

        self.ax.scatter(
            self.X_test, self.y_test, color='green', label='Test Samples'
        )

    def plot_pred_data(self) -> None:
        if self.X_pred is None or self.y_pred is None:
            return

        self.ax.scatter(
            self.X_pred, self.y_pred, color='cyan', label='Predicted Samples'
        )

    def get_domain(self) -> tuple[float, float]:
        """Returns the observed feature range across training and test
        samples.

        Returns:
            A tuple containing the minimum and maximum feature values observed
            across the training and test datasets.
        """

        X = self.X_train

        if self.X_test is not None:
            X = np.concatenate((X, self.X_test), axis=0)

        if self.X_pred is not None:
            X = np.concatenate((X, self.X_pred), axis=0)

        return X.min(), X.max()

    def get_curve_x_range(
        self, extension_rate: float = 0.05
    ) -> tuple[float, float]:
        """Returns the x-range used when plotting the model prediction curve.

        Returns:
            A tuple containing the minimum and maximum x-values used when
            plotting the model prediction curve.
        """

        x_min, x_max = self.get_domain()
        x_range = x_max - x_min

        return (
            x_min - extension_rate * x_range,
            x_max + extension_rate * x_range,
        )

    def plot_curve(self) -> None:
        """Plots the model prediction across the observed feature range."""

        x_min, x_max = self.get_curve_x_range()
        x_values = np.linspace(x_min, x_max, 512).reshape(-1, 1)
        y_values = self.model(x_values)

        self.ax.plot(
            x_values, y_values, color='orange', label=self.curve_description
        )

    def set_labels_legend(self) -> None:
        """Applies axis labels and legend entries from the dataset schema."""

        if self.schema.target_names is None or self.schema.target_units is None:
            raise ValueError(
                'Target names and units must be provided in the schema.'
            )

        feature_labels: list[str] = combine_names_and_units(
            self.schema.feature_names,
            self.schema.feature_units,
        )

        target_labels: list[str] = combine_names_and_units(
            self.schema.target_names,
            self.schema.target_units,
        )

        self.ax.set_xlabel(feature_labels[0])
        self.ax.set_ylabel(target_labels[0])
        self.ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        self.ax.xaxis.set_major_formatter(StrMethodFormatter('{x:.0f}'))
        self.ax.legend()
