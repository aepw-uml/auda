from typing import override

from auda.step.anomaly.isolation_forest import IsolationForest
from auda.step.evaluator.cross_validation import CrossValidationEvaluator
from auda.step.regression.support_vector_regression import (
    SupportVectorRegression,
)
from auda.step.spec import Spec
from auda.step.transformer.z_norm import ZNorm
from auda.step.tuner.multi_stage_random_search_tuner import (
    MultiStageRandomSearchTuner,
)
from auda.utils.pipeline import IOValueMap, Pipeline, Step, step


@step(
    id='HT-SVR',
    description='Hyperparameter tuning for Support Vector Regression (SVR)',
    input_specs=[
        Spec.ON.optional(Spec.DATASET.name),
        Spec.METRIC,
        Spec.EXPECT_HIGHER.optional(),
        Spec.SEED.optional(42),
        Spec.SEARCH_SPACE.optional([[(0.1, 100.0)], [(0.001, 1.0)]]),
        Spec.NUM_ITERATIONS.optional(100),
        Spec.ELITE_FRACTIONS.optional([0.1, 0.05]),
        Spec.REFINEMENT_WIDTHS.optional([[10, 0.2], [2, 0.05]]),
    ],
    output_specs=[
        Spec.BEST_SCORE,
        Spec.BEST_HYPERPARAMETERS,
        Spec.HYPERPARAMETERS_SCORE_LISTS,
    ],
)
class SvrTuner(Step):
    @override
    def run(
        self, seed: int, metric: str, expect_higher: bool | None
    ) -> IOValueMap:
        # ZNorm -> IsolationForest -> SVR
        cross_validation_pipeline = Pipeline(
            [
                ZNorm,
                IsolationForest,
                SupportVectorRegression,
            ],
            [
                {Spec.SEED.name: seed},
                {Spec.ON.name: Spec.NORMALIZED_DATASET.name},
                {Spec.ON.name: Spec.INLIER_DATASET.name},
                {Spec.ON.name: Spec.NORMALIZED_DATASET.name},
            ],
        )

        # CrossValidationEvaluator (ZNorm -> IsolationForest -> SVR)
        random_tuner_pipeline = Pipeline(
            [CrossValidationEvaluator],
            [{Spec.PIPE.name: cross_validation_pipeline, Spec.SEED.name: seed}],
        )

        pipeline = Pipeline(
            [MultiStageRandomSearchTuner],
            [
                {
                    **self._inputs,
                    Spec.PIPE.name: random_tuner_pipeline,
                    Spec.HYPERPARAMETER_NAMES.name: [
                        Spec.C.name,
                        Spec.EPSILON.name,
                    ],
                    Spec.HYPERPARAMETER_DOMAINS.name: [
                        (0.1, 100.0),  # C
                        (0.001, 1.0),  # Epsilon
                    ],
                }
            ],
        )

        pipeline.run()

        if expect_higher is None:
            expect_higher = metric.lower() == 'r2'

        hp_score_list = pipeline.get_value(
            Spec.HYPERPARAMETERS_SCORE_LIST.name
        )[:]
        hp_score_list.sort(key=lambda x: x[1], reverse=expect_higher)
        best_hp, best_score = hp_score_list[0]

        return {
            Spec.BEST_SCORE.name: best_score,
            Spec.BEST_HYPERPARAMETERS.name: best_hp,
            Spec.HYPERPARAMETERS_SCORE_LISTS.name: pipeline.get_value(
                Spec.HYPERPARAMETERS_SCORE_LISTS.name
            ),
        }
