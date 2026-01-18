from random import randrange
from typing import override

from auda.step.anomaly.isolation_forest import IsolationForest
from auda.step.model.ridge_regression import RidgeRegressionModel
from auda.step.spec import Spec
from auda.step.transformer.z_norm import ZNorm
from auda.step.tuner.msrs_automator import MsrsAutomator
from auda.utils.pipeline import IOValueMap, Pipeline, Step, step


@step(
    id='HT-RR',
    description='Hyperparameter tuning for Ridge Regression.',
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
class RidgeRegressionTuner(Step):
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
                RidgeRegressionModel,
                {Spec.ON.name: Spec.INLIER_DATASET.name},
            )
        else:
            train_pipe.append(
                RidgeRegressionModel,
                {Spec.ON.name: Spec.NORMALIZED_DATASET.name},
            )

        hyperparameter_names = [Spec.ALPHA.name]
        hyperparameter_domains = [(1e-6, 1e3)]
        pipeline = Pipeline().append(
            MsrsAutomator,
            {
                Spec.PIPE.name: train_pipe,
                Spec.HYPERPARAMETER_NAMES.name: hyperparameter_names,
                Spec.HYPERPARAMETER_DOMAINS.name: hyperparameter_domains,
                **self._inputs,
                Spec.SEED.name: seed,
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
            Spec.SEED.name: seed,
            Spec.HYPERPARAMETER_NAMES.name: hyperparameter_names,
            Spec.HYPERPARAMETER_DOMAINS.name: hyperparameter_domains,
        }
