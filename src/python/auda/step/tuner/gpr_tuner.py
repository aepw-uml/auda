from random import randrange
from typing import override

from auda.step.anomaly.isolation_forest import IsolationForest
from auda.step.model.gaussian_process_regression import (
    GaussianProcessRegressionModel,
)
from auda.step.spec import Spec
from auda.step.transformer.z_norm import ZNorm
from auda.step.tuner.msrs_automator import MsrsAutomator
from auda.utils.pipeline import IOValueMap, Pipeline, Step, step


@step(
    id='HT-GPR',
    description='Hyperparameter tuning for Gaussian Process Regression (GPR)',
    input_specs=[
        Spec.ON.optional(Spec.DATASET.name),
        Spec.METRIC,
        Spec.EXPECT_HIGHER.optional(),
        Spec.SEED.optional(),
        Spec.SEARCH_SPACE.optional([[(1e-3, 1e3)], [(1e-6, 1e1)]]),
        Spec.NUM_ITERATIONS.optional(50),
        Spec.ELITE_FRACTIONS.optional([0.1, 0.05]),
        Spec.REFINEMENT_WIDTHS.optional([[1.0, 0.5], [0.3, 0.2]]),
        Spec.USE_ANOMALY_DETECTION.optional(True),
    ],
    output_specs=[
        Spec.BEST_SCORE,
        Spec.BEST_HYPERPARAMETERS,
        Spec.HYPERPARAMETERS_SCORE_LISTS,
    ],
)
class GprTuner(Step):
    @override
    def run(
        self,
        seed: int | None,
        use_anomaly_detection: bool,
    ) -> IOValueMap:
        if seed is None:
            seed = randrange(2**32)

        train_pipe = Pipeline().append(ZNorm)
        if use_anomaly_detection:
            train_pipe = train_pipe.append(
                IsolationForest,
                {Spec.ON.name: Spec.NORMALIZED_DATASET.name},
            )
            train_pipe.append(
                GaussianProcessRegressionModel,
                {Spec.ON.name: Spec.INLIER_DATASET.name},
            )
        else:
            train_pipe.append(
                GaussianProcessRegressionModel,
                {Spec.ON.name: Spec.NORMALIZED_DATASET.name},
            )

        pipeline = Pipeline().append(
            MsrsAutomator,
            {
                **self._inputs,
                Spec.PIPE.name: train_pipe,
                Spec.HYPERPARAMETER_NAMES.name: [
                    Spec.LENGTH_SCALE.name,
                    Spec.NOISE_LEVEL.name,
                ],
                Spec.HYPERPARAMETER_DOMAINS.name: [
                    (1e-3, 1e3),
                    (1e-6, 1e1),
                ],
            },
        )
        pipeline.run()

        return {
            Spec.BEST_SCORE.name: pipeline.get_value(Spec.BEST_SCORE.name),
            Spec.BEST_HYPERPARAMETERS.name: pipeline.get_value(
                Spec.BEST_HYPERPARAMETERS.name
            ),
            Spec.HYPERPARAMETERS_SCORE_LISTS.name: pipeline.get_value(
                Spec.HYPERPARAMETERS_SCORE_LISTS.name
            ),
        }
