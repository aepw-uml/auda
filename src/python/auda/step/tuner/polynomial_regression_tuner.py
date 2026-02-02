from random import randrange
from typing import override

from auda.step.anomaly.isolation_forest import IsolationForest
from auda.step.dataset import DatasetBasedStep
from auda.step.evaluator.cross_validation import CrossValidationEvaluator
from auda.step.evaluator.time_series_cross_validation_evaluator import (
    TimeSeriesCrossValidationEvaluator,
)
from auda.step.model.polynomial_regression import (
    PolynomialRegressionModel,
)
from auda.step.spec import Dataset, Spec
from auda.step.transformer.z_norm import ZNorm
from auda.utils.pipeline import IOValueMap, Pipeline, step


@step(
    id='HT-PR',
    description='Hyperparameter tuning for Polynomial Regression.',
    input_specs=[
        Spec.ON.optional(Spec.DATASET.name),
        Spec.METRIC,
        Spec.EXPECT_HIGHER.optional(),
        Spec.SEED.optional(),
        Spec.NUM_ITERATIONS.optional(50),
        Spec.USE_ANOMALY_DETECTION.optional(True),
        Spec.USE_TIME_SERIES.optional(False),
    ],
    output_specs=[
        Spec.BEST_SCORE,
        Spec.BEST_HYPERPARAMETERS,
        Spec.SEED,
        Spec.HYPERPARAMETER_NAMES,
        Spec.HYPERPARAMETER_DOMAINS,
    ],
)
class PolynomialRegressionTuner(DatasetBasedStep):
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

        print('ds' + self.pipeline.get_value(Spec.USE_TIME_SERIES.name))

        if expect_higher is None:
            expect_higher = metric.lower() == 'r2'

        train_pipe = Pipeline().append(ZNorm, {Spec.ON.name: on})

        if use_anomaly_detection:
            train_pipe = train_pipe.append(
                IsolationForest,
                {Spec.ON.name: Spec.NORMALIZED_DATASET.name},
            )
            train_pipe.append(
                PolynomialRegressionModel,
                {Spec.ON.name: Spec.INLIER_DATASET.name},
            )
        else:
            train_pipe.append(
                PolynomialRegressionModel,
                {Spec.ON.name: Spec.NORMALIZED_DATASET.name},
            )

        if not use_time_series:
            eval_pipe = Pipeline().append(
                CrossValidationEvaluator,
                {Spec.PIPE.name: train_pipe, Spec.SEED.name: seed},
            )
        else:
            eval_pipe = Pipeline().append(
                TimeSeriesCrossValidationEvaluator,
                {Spec.PIPE.name: train_pipe, Spec.SEED.name: seed},
            )

        hp_score_list: list[tuple[list[float], float]] = []

        for degree in range(1, 13):
            eval_pipe.reset().run(
                {Spec.ON.name: on, Spec.DEGREE.name: degree, **self._inputs}
            )

            score = eval_pipe.get_value(metric.upper())
            if score is None:
                raise ValueError(
                    f'CrossValidationEvaluator did not produce a value for '
                    f'metric "{metric}".'
                )

            hp_score_list.append(([float(degree)], float(score)))

        hp_score_list.sort(key=lambda x: x[1], reverse=expect_higher)
        best_hp, best_score = hp_score_list[0]

        return {
            Spec.SEED.name: seed,
            Spec.BEST_SCORE.name: best_score,
            Spec.BEST_HYPERPARAMETERS.name: best_hp,
            Spec.HYPERPARAMETER_NAMES.name: [Spec.DEGREE.name],
            Spec.HYPERPARAMETER_DOMAINS.name: [(1, 12)],
        }
