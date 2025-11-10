from typing import List, override

from auda.dat.datasets import UnlabeledSamples
from auda.utils.pipeline import IOSpec, Task, task

from .__common import PLOTTER_KIND, PlotterISName, PlotterOSName


@task(
    id='PL-CM',
    kind=PLOTTER_KIND,
    description='Visualizes a correlation matrix as a heatmap.',
    input_specs={
        PlotterISName.FEATURE_NAMES: IOSpec(dtype=List[str]),
        PlotterISName.CORRELATION_MATRIX: IOSpec(dtype=UnlabeledSamples),
    },
    output_specs={
        PlotterOSName.FIGURE: IOSpec(dtype=object),
    },
)
class CorrelationMatrixPlotter(Task):
    @override
    def run(self) -> None:
        import textwrap

        import matplotlib.pyplot as plt
        import numpy as np
        import pandas as pd
        import seaborn as sns

        feature_names = self.get_input(PlotterISName.FEATURE_NAMES)
        corr = np.array(self.get_input(PlotterISName.CORRELATION_MATRIX))

        # ---- Validate correlation matrix
        if corr.shape[0] != corr.shape[1]:
            raise ValueError(
                f'Correlation matrix must be square; got shape {corr.shape}.'
            )
        if corr.shape[0] != len(feature_names):
            raise ValueError(
                f'Feature list length ({len(feature_names)}) must match correlation '
                f'matrix dimension ({corr.shape[0]}).'
            )

        # ---- Build a DataFrame for nicer axis labels
        df = pd.DataFrame(corr, index=feature_names, columns=feature_names)

        # ---- Mask the upper triangle (because the matrix is symmetric)
        mask = np.triu(np.ones_like(df, dtype=bool), k=1)

        # ---- Figure size that scales with number of features
        n = len(feature_names)
        fig_w = max(6.5, min(16.0, 0.55 * n + 2.0))
        fig_h = max(5.0, min(14.0, 0.55 * n + 2.0))

        def _wrap_label(s: str, width: int = 16) -> str:
            return '\n'.join(textwrap.wrap(s, width=width, break_long_words=False))

        wrapped_labels = [_wrap_label(s) for s in feature_names]

        plt.close('all')
        fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=200)

        sns.heatmap(
            df,
            mask=mask,
            ax=ax,
            cmap='PiYG',
            vmin=-1.0,
            vmax=1.0,
            center=0.0,
            square=False,
            cbar=True,
            cbar_kws={'label': '', 'shrink': 0.9},
            annot=True,
            fmt='.2f',
            annot_kws={'fontsize': 7},
            linewidths=0.5,
            linecolor='white',
        )

        # ---- Axes formatting
        ax.set_xticks(np.arange(n) + 0.5)
        ax.set_yticks(np.arange(n) + 0.5)

        # ---- Use wrapped labels; anchor rotation to avoid collisions
        ax.set_xticklabels(
            wrapped_labels,
            rotation=45,
            ha='right',
            va='center',
            rotation_mode='anchor',
            fontsize=7,
        )
        ax.set_yticklabels(wrapped_labels, rotation=0, va='center', fontsize=7)

        # Give a bit more room for multi-line ticks without changing the core parameters
        fig.tight_layout()
        fig.subplots_adjust(left=0.28, bottom=0.25)

        # ---- Populate outputs
        self.set_output(PlotterOSName.FIGURE, fig)
