from typing import override

import numpy as np
from dataset.dataset import DatasetSchema
from step.plot.plotter import Plotter
from step.tuner.random_search import HyperparameterScore


class MrsBestSoFar(Plotter):
    """Plots the running best score across multi-stage random search runs."""

    def __init__(
        self,
        hyperparameter_scores_list: list[list[HyperparameterScore]],
        metric_name: str = 'MAPE',
        minimize: bool = True,
        title: str = 'Multi-stage random search best-so-far',
    ) -> None:
        """Initializes the best-so-far plotter.

        Args:
            hyperparameter_scores_list: Score histories grouped by search stage.
            metric_name: Name of the metric shown on the y-axis.
            minimize: Whether lower metric values are better.
            title: Plot title.
        """

        super().__init__(
            schema=DatasetSchema(feature_names=[], feature_units=[]),
            title=title,
        )

        self.hyperparameter_scores_list: list[list[HyperparameterScore]] = (
            hyperparameter_scores_list
        )
        self.metric_name: str = metric_name
        self.minimize: bool = minimize

    @override
    def plot(self) -> None:
        """Plots the running best score over all evaluated configurations."""

        self.ax.clear()
        scores, stage_end_indices = self._flatten_scores()
        self.ax.set_title(self.title)
        self.ax.set_xlabel('Evaluation')
        self.ax.set_ylabel(f'Best validation score ({self.metric_name}) so far')

        if not scores:
            self.ax.text(
                0.5,
                0.5,
                'No evaluations',
                ha='center',
                va='center',
                transform=self.ax.transAxes,
            )
            self.ax.grid(True, which='both', linestyle=':', linewidth=0.5)
            return

        best_so_far = self._compute_best_so_far(scores)
        x = np.arange(1, len(best_so_far) + 1)
        plotted_best_so_far = np.where(
            np.isfinite(best_so_far), best_so_far, np.nan
        )

        self.ax.plot(x, plotted_best_so_far, label='Best so far')

        for end_idx in stage_end_indices[:-1]:
            self.ax.axvline(end_idx + 0.5, linestyle='--', linewidth=1)

        stage_starts = [1] + [e + 1 for e in stage_end_indices[:-1]]
        stage_ends = stage_end_indices
        for i, (s, e) in enumerate(zip(stage_starts, stage_ends), start=1):
            mid = (s + e) / 2.0
            self.ax.text(
                mid,
                1.01,
                f'Stage {i}',
                ha='center',
                va='bottom',
                transform=self.ax.get_xaxis_transform(),
            )

        self.ax.margins(x=0.01, y=0.10)
        self.ax.grid(True, which='both', linestyle=':', linewidth=0.5)
        self.ax.legend()

    def _flatten_scores(self) -> tuple[list[float], list[int]]:
        """Flattens stage-organized scores and records stage boundaries.

        Returns:
            A tuple containing all scores in evaluation order and the inclusive
            end index of each stage in one-based plotting coordinates.
        """

        scores: list[float] = []
        stage_end_indices: list[int] = []
        for hyperparameter_scores in self.hyperparameter_scores_list:
            for score, _ in hyperparameter_scores:
                scores.append(score)

            stage_end_indices.append(len(scores))

        return scores, stage_end_indices

    def _compute_best_so_far(self, scores: list[float]) -> np.ndarray:
        """Computes the running best score while ignoring invalid values.

        Args:
            scores: Score values in evaluation order.

        Returns:
            A NumPy array containing the best score observed up to each
            evaluation.
        """

        clean_scores = np.asarray(scores, dtype=float)
        invalid_value = np.inf if self.minimize else -np.inf
        clean_scores[~np.isfinite(clean_scores)] = invalid_value

        if self.minimize:
            return np.minimum.accumulate(clean_scores)

        return np.maximum.accumulate(clean_scores)
