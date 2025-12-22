from typing import cast, override

import numpy as np
from auda.step.dataset import DatasetBasedStep, DatasetSchema
from auda.step.plot import PlotStep
from auda.step.spec import Dataset, Spec
from auda.utils.pipeline import IOValueMap, step


@step(
    id='MD-CM',
    description='Computes the correlation matrix among input features.',
    input_specs=[Spec.ON.optional(Spec.DATASET.name)],
    output_specs=[Spec.CORRELATION_MATRIX],
)
class CorrelationMatrix(DatasetBasedStep):
    @override
    def run(self, on: str | Dataset) -> IOValueMap:
        X = cast(np.ndarray, self.get_dataset_from_on(on))
        n = X.shape[0]

        # ---- Center data
        X_mean = np.mean(X, axis=0)
        X = X - X_mean

        covariance_matrix: np.ndarray = (X.T @ X) / (n - 1)
        std_dev = np.sqrt(np.diag(covariance_matrix))
        D_inv = np.diag(1 / std_dev)
        correlation_matrix: np.ndarray = D_inv @ covariance_matrix @ D_inv

        return {Spec.CORRELATION_MATRIX.name: correlation_matrix}


@step(
    id='PL-CM',
    description='Generates a heatmap plot of the correlation matrix.',
    input_specs=[
        Spec.CORRELATION_MATRIX,
        Spec.DATASET_SCHEMA,
    ],
    output_specs=[Spec.FIGURE, Spec.AXES],
)
class CorrelationMatrixPlotter(PlotStep):
    @override
    def run(
        self, correlation_matrix: np.ndarray, dataset_schema: DatasetSchema
    ) -> IOValueMap:
        import textwrap

        import matplotlib.pyplot as plt
        import numpy as np
        import pandas as pd
        import seaborn as sns

        corr = correlation_matrix
        feature_names = dataset_schema.feature_names

        # ---- Validate correlation matrix
        if corr.shape[0] != corr.shape[1]:
            raise ValueError(
                f'Correlation matrix must be square; got shape {corr.shape}.'
            )
        if corr.shape[0] != len(feature_names):
            raise ValueError(
                f'Feature list length ({len(feature_names)}) must match '
                f'correlation matrix dimension ({corr.shape[0]}).'
            )

        # ---- Build a DataFrame for nicer axis labels
        df = pd.DataFrame(corr)

        # ---- Mask the upper triangle (because the matrix is symmetric)
        mask = np.triu(np.ones_like(df, dtype=bool), k=1)

        # ---- Figure size that scales with number of features
        n = len(feature_names)
        fig_width = max(6.5, min(16.0, 0.55 * n + 2.0))
        fig_height = max(5.0, min(14.0, 0.55 * n + 2.0))

        def _wrap_label(s: str, width: int = 16) -> str:
            return '\n'.join(
                textwrap.wrap(s, width=width, break_long_words=False)
            )

        wrapped_labels = [_wrap_label(s) for s in feature_names]

        plt.close('all')
        fig, ax = plt.subplots(figsize=(fig_width, fig_height), dpi=200)

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

        # Give a bit more room for multi-line ticks without changing the core
        # parameters
        fig.tight_layout()
        fig.subplots_adjust(left=0.28, bottom=0.25)

        # ---- Populate outputs
        return self.regular_output(fig, ax)
