from random import randrange
from typing import override

from auda.step.anomaly.isolation_forest import IsolationForest
from auda.step.dataset import DatasetBasedStep
from auda.step.model.ridge_regression import PolynomialRidgeRegressionModel
from auda.step.spec import Dataset, Spec
from auda.step.transformer.z_norm import ZNorm
from auda.step.tuner.msrs_automator import MsrsAutomator
from auda.utils.pipeline import IOValueMap, Pipeline, step


@step(
    id='HT-PRR',
    description='Hyperparameter tuning for Polynomial Ridge Regression.',
    input_specs=[
        Spec.ON.optional(Spec.DATASET.name),
        Spec.METRIC,
        Spec.EXPECT_HIGHER.optional(),
        Spec.SEED.optional(),
        Spec.SEARCH_SPACE.optional([[(1e-6, 1e3)]]),
        Spec.NUM_ITERATIONS.optional(50),
        Spec.ELITE_FRACTIONS.optional([0.12, 0.06]),
        Spec.REFINEMENT_WIDTHS.optional([[0.5], [0.1]]),
        Spec.USE_ANOMALY_DETECTION.optional(True),
        Spec.USE_TIME_SERIES.optional(False),
    ],
    output_specs=[
        Spec.BEST_SCORE,
        Spec.BEST_HYPERPARAMETERS,
        Spec.HYPERPARAMETERS_SCORE_LISTS,
        Spec.SEED,
        Spec.HYPERPARAMETER_NAMES,
        Spec.HYPERPARAMETER_DOMAINS,
    ],
)
class PolynomialRidgeRegressionTuner(DatasetBasedStep):
    @override
    def run(
        self,
        on: str | Dataset,
        metric: str,
        expect_higher: bool | None,
        seed: int | None,
        use_anomaly_detection: bool,
        use_time_series: bool,
    ) -> IOValueMap:
        if seed is None:
            seed = randrange(2**32)

        if expect_higher is None:
            expect_higher = metric.lower() == 'r2'

        hyperparameter_names = [Spec.DEGREE.name, Spec.ALPHA.name]
        hyperparameter_domains = [(1, 12), (1e-6, 1e3)]

        best_score: float | None = None
        best_hyperparameters: list[float] | None = None
        best_hp_score_lists: list[list[tuple[list[float], float]]] = []

        for degree in range(1, 13):
            train_pipe = Pipeline().append(ZNorm, {Spec.ON.name: on})
            if use_anomaly_detection:
                train_pipe = train_pipe.append(
                    IsolationForest,
                    {Spec.ON.name: Spec.NORMALIZED_DATASET.name},
                )
                train_pipe.append(
                    PolynomialRidgeRegressionModel,
                    {
                        Spec.ON.name: Spec.INLIER_DATASET.name,
                        Spec.DEGREE.name: degree,
                    },
                )
            else:
                train_pipe.append(
                    PolynomialRidgeRegressionModel,
                    {
                        Spec.ON.name: Spec.NORMALIZED_DATASET.name,
                        Spec.DEGREE.name: degree,
                    },
                )

            pipeline = Pipeline().append(
                MsrsAutomator,
                {
                    Spec.PIPE.name: train_pipe,
                    Spec.HYPERPARAMETER_NAMES.name: [Spec.ALPHA.name],
                    Spec.HYPERPARAMETER_DOMAINS.name: [(1e-6, 1e3)],
                    **self._inputs,
                    Spec.SEED.name: seed,
                    Spec.USE_TIME_SERIES.name: use_time_series,
                },
            )
            pipeline.run()

            degree_score = pipeline.get_value(Spec.BEST_SCORE.name)
            best_alpha = pipeline.get_value(Spec.BEST_HYPERPARAMETERS.name)[0]
            hp_score_lists = pipeline.get_value(
                Spec.HYPERPARAMETERS_SCORE_LISTS.name
            )

            converted_lists: list[list[tuple[list[float], float]]] = []
            if hp_score_lists is not None:
                for stage_list in hp_score_lists:
                    converted_lists.append(
                        [
                            ([float(degree), float(hp[0])], float(score))
                            for hp, score in stage_list
                        ]
                    )

            if best_score is None:
                best_score = float(degree_score)
                best_hyperparameters = [float(degree), float(best_alpha)]
                best_hp_score_lists = converted_lists
            else:
                is_better = (
                    float(degree_score) > best_score
                    if expect_higher
                    else float(degree_score) < best_score
                )
                if is_better:
                    best_score = float(degree_score)
                    best_hyperparameters = [float(degree), float(best_alpha)]
                    best_hp_score_lists = converted_lists

        if best_hyperparameters is None or best_score is None:
            raise ValueError('No valid hyperparameter set was evaluated.')

        return {
            Spec.SEED.name: seed,
            Spec.BEST_SCORE.name: best_score,
            Spec.BEST_HYPERPARAMETERS.name: best_hyperparameters,
            Spec.HYPERPARAMETERS_SCORE_LISTS.name: best_hp_score_lists,
            Spec.HYPERPARAMETER_NAMES.name: hyperparameter_names,
            Spec.HYPERPARAMETER_DOMAINS.name: hyperparameter_domains,
        }
