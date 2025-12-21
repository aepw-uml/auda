from typing import List, Tuple, override

from auda.step.anomaly.isolation_forest import IsolationForest
from auda.step.evaluator.cross_validation import CrossValidationEvaluator
from auda.step.regression.support_vector_regression import (
    SupportVectorRegression,
)
from auda.step.spec import Spec
from auda.step.transformer.z_norm import ZNorm
from auda.step.tuner.random_tuner import RandomTuner
from auda.utils.pipeline import IOValueMap, Pipeline, Step, step


@step(
    id='HT-SVR',
    description='Hyperparameter tuning for Support Vector Regression (SVR)',
    input_specs=[
        Spec.ON.optional(Spec.DATASET.name),
        Spec.METRIC,
        Spec.EXPECT_HIGHER.optional(),
        Spec.SEED.optional(42),
        Spec.SAMPLING_INTERVALS.optional([[(0.01, 100.0)], [(0.001, 1.0)]]),
    ],
    output_specs=[Spec.BEST_SCORE, Spec.BEST_HYPERPARAMETERS],
)
class SvrTuner(Step):
    @override
    def run(self, seed: int) -> IOValueMap:
        # Stage 0 (Coarse Search)
        hp_score_list = self.get_hyperparameters_score_list(seed)
        best_hp, best_score = hp_score_list[0]

        return {
            Spec.BEST_SCORE.name: best_score,
            Spec.BEST_HYPERPARAMETERS.name: best_hp,
        }

    def get_hyperparameters_score_list(
        self, seed: int
    ) -> List[Tuple[List[float], float]]:
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
            ],
        )
        random_tuner_pipeline = Pipeline(
            [CrossValidationEvaluator],
            [{Spec.PIPE.name: cross_validation_pipeline, Spec.SEED.name: seed}],
        )

        pipeline = Pipeline(
            [RandomTuner],
            [
                {
                    **self._inputs,
                    Spec.PIPE.name: random_tuner_pipeline,
                    Spec.HYPERPARAMETER_NAMES.name: [
                        Spec.C.name,
                        Spec.EPSILON.name,
                    ],
                }
            ],
        )

        pipeline.run()

        return pipeline.get_value(Spec.HYPERPARAMETERS_SCORE_LIST.name)
