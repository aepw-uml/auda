import math
from typing import List, Tuple, override

import numpy as np
from auda.core import auda
from auda.step.plot import PlotStep
from auda.step.spec import Interval, Spec
from auda.step.tuner.random_search_tuner import RandomSearchTuner
from auda.utils.pipeline import IOValueMap, Pipeline, step


@step(
    id='HT-MSRS',
    description='Multi-stage random search hyperparameter tuner.',
    input_specs=[
        Spec.ON.optional(Spec.DATASET.name),
        Spec.METRIC,
        Spec.EXPECT_HIGHER.optional(),
        Spec.SEED.optional(42),
        Spec.SEARCH_SPACE,
        Spec.NUM_ITERATIONS.optional(100),
        Spec.PIPE,
        Spec.HYPERPARAMETER_NAMES,
        Spec.ELITE_FRACTIONS,
        Spec.REFINEMENT_WIDTHS,
        Spec.HYPERPARAMETER_DOMAINS,
    ],
    output_specs=[
        Spec.HYPERPARAMETERS_SCORE_LIST,
        Spec.HYPERPARAMETERS_SCORE_LISTS,
    ],
)
class MultiStageRandomSearchTuner(RandomSearchTuner):
    @override
    def run(
        self,
        search_space: List[List[Interval]],
        elite_fractions: List[float],
        refinement_widths: List[List[float]],
        hyperparameter_domains: List[Interval],
    ) -> IOValueMap:
        logger = auda.get_logger(__class__.__name__)

        num_stages = len(elite_fractions)

        if len(refinement_widths) != num_stages:
            raise ValueError(
                'Length of refinement_widths must match length of '
                'elite_fractions.'
            )

        hp_score_lists: List[List[Tuple[List[float], float]]] = []
        hp_score_list: List[Tuple[List[float], float]] = []
        for stage_idx in range(num_stages + 1):
            logger.info(
                f'Starting stage {stage_idx + 1} of {num_stages + 1}...'
            )

            if stage_idx == num_stages:
                hp_score_list, _ = self.stage(search_space)
            else:
                hp_score_list, search_space = self.stage(
                    search_space,
                    elite_fractions[stage_idx],
                    refinement_widths[stage_idx],
                    hyperparameter_domains,
                )

            hp_score_lists.append(hp_score_list)

        return {
            Spec.HYPERPARAMETERS_SCORE_LIST.name: hp_score_list,
            Spec.HYPERPARAMETERS_SCORE_LISTS.name: hp_score_lists,
        }

    def stage(
        self,
        search_space: List[List[Interval]],
        elite_fraction: float | None = None,
        refinement_widths: List[float] | None = None,
        hyperparameter_domains: List[Interval] | None = None,
    ) -> Tuple[List[Tuple[List[float], float]], List[List[Interval]]]:
        """Performs a single stage of random search tuning.

        Args:
            search_space: Current search space for hyperparameters.
            elite_fraction: Fraction of top candidates to consider as elite.
            refinement_widths: Widths for refining the search space around elite
                candidates.
            hyperparameter_domains: Overall domains for each hyperparameter.

        Returns:
            A tuple containing:
                - A list of tuples with hyperparameter configurations and their
                    scores.
                - A list of refined search spaces for the next stage.
        """

        num_hyperparameters = len(search_space)

        pipeline = Pipeline(
            [RandomSearchTuner],
            [{**self._inputs, Spec.SEARCH_SPACE.name: search_space}],
        )
        pipeline.run()
        hp_score_list = pipeline.get_value(Spec.HYPERPARAMETERS_SCORE_LIST.name)

        expect_higher = self.get_input(Spec.EXPECT_HIGHER.name)
        if expect_higher is None:
            expect_higher = self.get_input(Spec.METRIC.name).lower() == 'r2'

        unsorted_hp_score_list = hp_score_list.copy()
        hp_score_list.sort(key=lambda x: x[1], reverse=expect_higher)

        if elite_fraction and refinement_widths and hyperparameter_domains:
            num_elite_candidates = max(
                1, int(len(hp_score_list) * elite_fraction)
            )
            elite_candidates: List[List[float]] = [
                hp_score_entry[0]
                for hp_score_entry in hp_score_list[:num_elite_candidates]
            ]

            next_search_space: List[List[Interval]] = [
                [] for _ in range(num_hyperparameters)
            ]
            for candidate in elite_candidates:
                for i in range(len(candidate)):
                    hp = candidate[i]
                    refinement_width = refinement_widths[i]
                    minimum = hyperparameter_domains[i][0]
                    maximum = hyperparameter_domains[i][1]
                    next_search_space[i].append(
                        (
                            max(hp - refinement_width, minimum),
                            min(hp + refinement_width, maximum),
                        )
                    )

        else:
            next_search_space = search_space

        return unsorted_hp_score_list, next_search_space


@step(
    id='PL-MSRS-BSF',
    description='Plot the best score for MSRS tuner over all stages.',
    input_specs=[
        Spec.HYPERPARAMETERS_SCORE_LISTS,
    ],
    output_specs=[Spec.FIGURE, Spec.AXES],
)
class BestSoFar(PlotStep):
    @override
    def run(
        self, hyperparameters_score_lists: List[List[Tuple[List[float], float]]]
    ) -> IOValueMap:
        # ---- Flatten scores, and record where each stage ends
        scores: List[float] = []
        stage_end_indices: List[int] = []
        for hp_score_list in hyperparameters_score_lists:
            for _, score in hp_score_list:
                scores.append(float(score))

            stage_end_indices.append(len(scores))

        figure, axes = self.create_plot_or_default()

        if not scores:
            axes.set_title('Best-so-far (no evaluations)')
            axes.set_xlabel('Evaluation')
            axes.set_ylabel('Best score so far')

            return {Spec.FIGURE.name: figure, Spec.AXES.name: axes}

        # Handle NaNs by treating them as -inf (so they never become "best")
        clean_scores = np.array(
            [
                s if (s is not None and not math.isnan(s)) else -np.inf
                for s in scores
            ],
            dtype=float,
        )

        # Best-so-far = cumulative maximum
        # best_so_far = np.maximum.accumulate(clean_scores)
        best_so_far = np.minimum.accumulate(clean_scores)

        x = np.arange(1, len(best_so_far) + 1)  # 1-based evaluation index

        axes.plot(x, best_so_far, marker=None)
        axes.set_xlabel('Evaluation')
        axes.set_ylabel('Best validation score (MAPE) so far')
        # axes.set_title('MSRS tuning convergence (best-so-far)')

        # Draw stage boundaries (skip the final boundary at N to avoid a line
        # at the edge)
        for end_idx in stage_end_indices[:-1]:
            axes.axvline(end_idx + 0.5, linestyle='--', linewidth=1)

        stage_starts = [1] + [e + 1 for e in stage_end_indices[:-1]]
        stage_ends = stage_end_indices
        y_top = (
            float(np.nanmax(best_so_far[np.isfinite(best_so_far)]))
            if np.isfinite(best_so_far).any()
            else 0.0
        )
        for i, (s, e) in enumerate(zip(stage_starts, stage_ends), start=1):
            mid = (s + e) / 2.0
            axes.text(mid, y_top, f'Stage {i}', ha='center', va='bottom')

        axes.margins(x=0.01, y=0.10)
        axes.grid(True, which='both', linestyle=':', linewidth=0.5)

        return {
            Spec.FIGURE.name: figure,
            Spec.AXES.name: axes,
        }
