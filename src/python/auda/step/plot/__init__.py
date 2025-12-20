from typing import Callable, List

import numpy as np
from auda.utils.pipeline import Step


class PlotStep(Step):
    def check_feature_dimension(
        self,
        X: np.ndarray,
        expected_dimension: int | None = None,
        checker: Callable[[int], bool] | None = None,
    ) -> None:
        """Check the feature dimension against expected dimension or custom
        checker.

        Args:
            expected_dimension: The expected feature dimension.
            checker: A custom function to validate the feature dimension.
        """

        feature_dimension = X.shape[1]

        if expected_dimension and feature_dimension != expected_dimension:
            raise ValueError(
                f'Expected feature dimension {expected_dimension}, but got '
                f'{feature_dimension} for step {self.spec.id}'
            )

        if checker and not checker(feature_dimension):
            raise ValueError(
                f'Invalid feature dimension: {feature_dimension} for step '
                f'{self.spec.id}'
            )

    def create_plot_or_default(self, fig=None, ax=None):
        """Create a matplotlib figure and axes."""
        import matplotlib.pyplot as plt

        if fig and ax:
            return fig, ax

        return plt.subplots()

    def combine_names_units(
        self, names: List[str], units: List[str]
    ) -> List[str]:
        """Combine names and units into formatted strings.

        Args:
            names: List of names.
            units: List of units.

        Returns:
            List of formatted strings combining names and units.
        """

        return [
            f'{name} ({unit})' if unit else name
            for name, unit in zip(names, units)
        ]

    def regular_output(self, fig, ax) -> dict:
        """Prepare the regular output dictionary.

        Args:
            figure: The matplotlib figure.
            ax: The matplotlib axes.

        Returns:
            A dictionary containing the figure and axes.
        """

        from auda.step.spec import Spec

        return {Spec.FIGURE.name: fig, Spec.AXES.name: ax}
