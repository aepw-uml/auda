from typing import List, Tuple, override

from auda.core import auda
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

        return {Spec.HYPERPARAMETERS_SCORE_LIST.name: hp_score_list}

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

        return hp_score_list, next_search_space
