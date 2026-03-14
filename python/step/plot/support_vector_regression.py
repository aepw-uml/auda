from typing import cast, override

import numpy as np
from step.plot.plotter import RegressionPlotter


class SupportVectorRegressionPlotter(RegressionPlotter):
    """Plots support vector regression results."""

    @override
    def plot(self) -> None:
        super().plot()

        self.plot_epsilon_tube()
        self.plot_support_vectors()
        self.curve_description = 'Support Vector Regression'
        self.ax.set_title('Support Vector Regression')

        self.set_labels_legend()

    def plot_epsilon_tube(self) -> None:
        """Plots the epsilon tube around the regression curve."""

        epsilon_optional = cast(
            float | None, self.hyperparameters.get('epsilon')
        )
        if epsilon_optional is None:
            raise ValueError(
                'Epsilon hyperparameter is required for support vector '
                'regression.'
            )

        epsilon: float = epsilon_optional
        y_std: float = self.y_train.std()
        epsilon = epsilon * y_std

        x_min, x_max = self.get_domain()
        x_values = np.linspace(x_min, x_max, 512).reshape(-1, 1)
        y_values = self.model(x_values)

        self.ax.fill_between(
            x_values.ravel(),
            y_values - epsilon,
            y_values + epsilon,
            color='orange',
            alpha=0.25,
            label='ε-tube',
            zorder=2,
        )

    def plot_support_vectors(self) -> None:
        """Plots the support vectors."""

        support_vector_indices = self.parameters.get('support_vector_indices')
        if support_vector_indices is None:
            raise ValueError(
                'Support vector indices are required to plot support vectors.'
            )

        x_support_vectors = self.X_train[support_vector_indices]
        y_support_vectors = self.y_train[support_vector_indices]
        self.ax.scatter(
            x_support_vectors,
            y_support_vectors,
            color='cornflowerblue',
            label='Support Vectors',
            edgecolor='black',
            zorder=3,
        )
