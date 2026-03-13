import os
from abc import ABC, abstractmethod
from typing import Callable, override

import matplotlib.pyplot as plt
import numpy as np
from dataset.dataset import DatasetSchema
from matplotlib.axes import Axes
from matplotlib.figure import Figure


class Plotter(ABC):
    def __init__(self, schema: DatasetSchema) -> None:
        self.schema: DatasetSchema = schema

        # Initialize the figure and axes for plotting.
        fig, ax = plt.subplots()
        self.fig: Figure = fig
        self.ax: Axes = ax

    @abstractmethod
    def plot(self) -> None:
        pass

    def show(self) -> None:
        self.fig.tight_layout()
        plt.show()

    def save(self, file_path: str) -> None:
        self.fig.tight_layout()

        os.makedirs(os.path.dirname(file_path) or '.', exist_ok=True)
        self.fig.savefig(
            file_path,
            bbox_inches='tight',
            pad_inches=0.1,
        )


class RegressionPlotter(Plotter, ABC):
    @override
    def __init__(
        self,
        schema: DatasetSchema,
        model: Callable[[np.ndarray], np.ndarray],
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray,
    ) -> None:
        super().__init__(schema)

        self.model: Callable[[np.ndarray], np.ndarray] = model
        self.X_train: np.ndarray = X_train
        self.y_train: np.ndarray = y_train
        self.X_test: np.ndarray = X_test
        self.y_test: np.ndarray = y_test
        self.schema: DatasetSchema = schema

    @override
    def plot(self) -> None:
        self.plot_training_data()
        self.plot_test_data()
        self.plot_curve()

    def plot_training_data(self) -> None:
        self.ax.scatter(
            self.X_train, self.y_train, color='blue', label='Training Samples'
        )

    def plot_test_data(self) -> None:
        self.ax.scatter(
            self.X_test, self.y_test, color='orange', label='Test Samples'
        )

    def plot_curve(self) -> None:
        X = np.concatenate((self.X_train, self.X_test), axis=0)
        x_min, x_max = X.min(), X.max()

        x_values = np.linspace(x_min, x_max, 512).reshape(-1, 1)
        y_values = self.model(x_values)

        self.ax.plot(x_values, y_values, color='red', label='Model Prediction')

    def set_legend(self) -> None:
        if self.schema.target_names is None or self.schema.target_units is None:
            raise ValueError(
                'Target names and units must be provided in the schema.'
            )

        self.ax.set_xlabel(self.schema.feature_names[0])
        self.ax.set_ylabel(self.schema.target_names[0])
        self.ax.legend()
