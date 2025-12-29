from auda.step.dataset import DatasetBasedStep
from auda.step.evaluator.cross_validation import CrossValidationEvaluator
from auda.step.spec import Interval, Spec
from auda.step.tuner.multi_stage_random_search_tuner import (
    MultiStageRandomSearchTuner,
)
from auda.utils.pipeline import IOValueMap, Pipeline, step


@step(
    id='HT-MSRS-AUTOMATOR',
    description='Automates MSRS.',
    input_specs=[
        Spec.ON.optional(Spec.DATASET.name),
        Spec.PIPE.desc('The train pipeline.'),
        Spec.HYPERPARAMETER_NAMES,
        Spec.HYPERPARAMETER_DOMAINS,
        Spec.METRIC,
        Spec.EXPECT_HIGHER.optional(),
        Spec.SEED.optional(42),
        Spec.SEARCH_SPACE,
        Spec.NUM_ITERATIONS.optional(100),
        Spec.ELITE_FRACTIONS,
        Spec.REFINEMENT_WIDTHS,
        Spec.USE_ANOMALY_DETECTION.optional(True),
    ],
    output_specs=[
        Spec.BEST_SCORE,
        Spec.BEST_HYPERPARAMETERS,
        Spec.HYPERPARAMETERS_SCORE_LISTS,
    ],
)
class MsrsAutomator(DatasetBasedStep):
    def run(
        self,
        seed: int | None,
        pipe: Pipeline,
        metric: str,
        expect_higher: bool | None,
        hyperparameter_names: list[str],
        hyperparameter_domains: list[Interval],
    ) -> IOValueMap:
        # Cross-validation wrapper (this produces MAE/RMSE/R2/MAPE)
        cross_validation_pipe = Pipeline().append(
            CrossValidationEvaluator,
            {Spec.PIPE.name: pipe, Spec.SEED.name: seed},
        )

        # Tune length_scale + noise_level using MSRS
        pipeline = Pipeline().append(
            MultiStageRandomSearchTuner,
            {
                **self._inputs,
                Spec.PIPE.name: cross_validation_pipe,
                Spec.HYPERPARAMETER_NAMES.name: hyperparameter_names,
                Spec.HYPERPARAMETER_DOMAINS.name: hyperparameter_domains,
            },
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
