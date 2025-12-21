from random import Random
from typing import List, Literal, Tuple, override

from auda.core import auda
from auda.step import create_pipeline_from_pipe
from auda.step.dataset import DatasetBasedStep
from auda.step.spec import Dataset, Interval, Spec
from auda.utils.pipeline import (
    IOValueMap,
    Pipeline,
    step,
)


@step(
    id='HT-RT',
    description='Randomly tunes hyperparameters for anomaly detection models.',
    input_specs=[
        Spec.ON.optional(Spec.DATASET.name),
        Spec.METRIC,
        Spec.EXPECT_HIGHER.optional(),
        Spec.SEED.optional(42),
        Spec.SAMPLING_INTERVALS,
        Spec.NUM_ITERATIONS.optional(100),
        Spec.PIPE,
        Spec.HYPERPARAMETER_NAMES,
    ],
    output_specs=[
        Spec.HYPERPARAMETERS_SCORE_LIST,
    ],
)
class RandomTuner(DatasetBasedStep):
    @override
    def run(
        self,
        on: str | Dataset,
        metric: Literal['mae', 'rmse', 'r2', 'mape'],
        expect_higher: bool | None,
        seed: int,
        sampling_intervals: List[List[Interval]],
        num_iterations: int,
        pipe: str | Pipeline,
        hyperparameter_names: List[str],
    ) -> IOValueMap:
        logger = auda.get_logger(__class__.__name__)

        if expect_higher is None:
            expect_higher = metric == 'r2'

        dataset = self.get_dataset_from_on(on)
        rng = Random(seed)
        num_hp = len(sampling_intervals)

        pipeline = create_pipeline_from_pipe(pipe)

        hp_score_list: List[Tuple[List[float], float]] = []
        for i in range(num_iterations):
            hp_values = [
                self._select_random_hyperparameter(intervals, rng)
                for intervals in sampling_intervals
            ]

            hp_map = {
                hyperparameter_names[i]: hp_values[i] for i in range(num_hp)
            }

            pipeline.reset().run({Spec.DATASET.name: dataset, **hp_map})
            score = pipeline.get_value(metric.upper())
            if score is None:
                raise ValueError(
                    f'Pipeline did not produce a value for metric "{metric}".'
                )

            hp_score_list.append((hp_values, score))

            hp_names_values_str = ', '.join(
                [
                    f'{hp_name}={hp_value:.3f}'
                    for hp_name, hp_value in zip(
                        hyperparameter_names, hp_values
                    )
                ]
            )

            logger.info(
                f'[{i + 1}/{num_iterations}] Evaluated '
                f'{hp_names_values_str}: score={score:.3f}'
            )

        hp_score_list.sort(key=lambda item: item[1], reverse=expect_higher)

        return {Spec.HYPERPARAMETERS_SCORE_LIST.name: hp_score_list}

    def _select_random_hyperparameter(
        self, intervals: List[Interval], rng: Random
    ) -> float:
        """Selects a random hyperparameter value from the given intervals.

        Args:
            intervals: List of intervals to choose from.
            rng: Random number generator.

        Returns:
            A randomly selected hyperparameter value.
        """

        selected_interval_index = rng.randint(0, len(intervals) - 1)
        selected_interval = intervals[selected_interval_index]

        return rng.uniform(selected_interval[0], selected_interval[1])
