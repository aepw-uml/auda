import textwrap
from typing import override

import numpy as np
from common.dataset import DatasetSchema
from matplotlib.ticker import StrMethodFormatter
from step.plot.plotter import Plotter


class FeatureImportancesPlotter(Plotter):
    """Plots ranked feature importance values for a fitted model."""

    @override
    def __init__(self, schema: DatasetSchema, title: str = '') -> None:
        """Initializes the feature importance plotter.

        Args:
            schema: Schema metadata used to label the plot.
            title: Optional plot title override.
        """

        super().__init__(schema, title)

    @override
    def plot(self, feature_importances: np.ndarray) -> None:
        """Plots the feature importances as a ranked horizontal bar chart.

        Args:
            feature_importances: Importance values aligned with the feature
                names in the dataset schema.
        """

        feature_importances = np.asarray(feature_importances, dtype=float)
        feature_names = np.asarray(self.schema.feature_names, dtype=object)

        if feature_importances.ndim != 1:
            raise ValueError(
                'Feature importances must be a 1D array; '
                f'got shape {feature_importances.shape}.'
            )
        if feature_importances.size == 0:
            raise ValueError('Feature importances cannot be empty.')
        if feature_importances.size != len(self.schema.feature_names):
            raise ValueError(
                f'Feature importances length ({feature_importances.size}) must '
                f'match the number of features '
                f'({len(self.schema.feature_names)}).'
            )

        sorted_indices = np.argsort(feature_importances)[::-1]
        sorted_importances = feature_importances[sorted_indices]
        sorted_feature_names = feature_names[sorted_indices]

        wrapped_feature_names = [
            '\n'.join(
                textwrap.wrap(
                    str(feature_name), width=22, break_long_words=False
                )
            )
            for feature_name in sorted_feature_names
        ]

        figure_height = max(4.5, 0.75 * sorted_importances.size + 1.5)
        self.fig.set_size_inches(10.0, figure_height)

        bar_colors = ['#1f4e79'] * sorted_importances.size
        if bar_colors:
            bar_colors[0] = '#0b2e4f'

        bars = self.ax.barh(
            wrapped_feature_names,
            sorted_importances,
            color=bar_colors,
            edgecolor='#16324f',
            linewidth=0.8,
            height=0.68,
        )
        self.ax.invert_yaxis()

        max_importance = float(sorted_importances.max())
        padding = max(0.01, max_importance * 0.12)
        self.ax.set_xlim(0.0, max_importance + padding)
        self.ax.set_xlabel('Importance Score')
        self.ax.set_ylabel('Feature')
        self.ax.set_title(self.title or 'Feature Importances')
        self.ax.xaxis.set_major_formatter(StrMethodFormatter('{x:.3f}'))
        self.ax.grid(axis='x', linestyle='--', linewidth=0.7, alpha=0.35)
        self.ax.set_axisbelow(True)
        self.ax.spines['top'].set_visible(False)
        self.ax.spines['right'].set_visible(False)

        for bar, importance in zip(bars, sorted_importances):
            self.ax.text(
                bar.get_width() + padding * 0.18,
                bar.get_y() + bar.get_height() / 2.0,
                f'{importance:.4f}',
                va='center',
                ha='left',
                fontsize=9,
                color='#16324f',
            )
