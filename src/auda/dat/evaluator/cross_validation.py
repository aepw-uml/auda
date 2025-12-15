from random import Random
from typing import List, Tuple, override

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from auda.dat import run_pipeline
from auda.dat.datasets import LabeledSamples
from auda.dat.models import ModelOSName
from auda.dat.models.__common import ModelISName
from auda.dat.stats.__common import StatOSName
from auda.utils.pipeline import IOSpec, Task, task

from .__common import EVALUATOR_KIND, EvaluatorISName, EvaluatorOSName


@task(
    id='CROSS-VALIDATION',
    kind=EVALUATOR_KIND,
    description='Performs k-fold cross-validation (k=4) to evaluate the performance of '
    'a regression model pipeline.',
    input_specs={
        EvaluatorISName.SAMPLES: IOSpec(dtype=LabeledSamples),
        EvaluatorISName.TRAINING_PIPE: IOSpec(dtype=str),
    },
    output_specs={
        EvaluatorOSName.MAE: IOSpec(dtype=float),
        EvaluatorOSName.RMSE: IOSpec(dtype=float),
        EvaluatorOSName.R2: IOSpec(dtype=float),
        EvaluatorOSName.MAPE: IOSpec(dtype=float),
    },
)
class CrossValidationEvaluator(Task):
    @override
    def run(self) -> None:
        pipeline_str = self.get_input(EvaluatorISName.TRAINING_PIPE)
        task_id_list = pipeline_str.split(' ')

        samples: LabeledSamples = self.get_input(EvaluatorISName.SAMPLES)
        Random(40).shuffle(samples)

        n = len(samples)
        k = 5

        # Split indices into k folds (keeps all samples, even if n % k != 0)
        folds = np.array_split(np.arange(n), k)

        maes: List[float] = []
        rmses: List[float] = []
        r2s: List[float] = []
        mapes: List[float] = []

        inputs = {**self._inputs}
        del inputs[EvaluatorISName.SAMPLES]

        for fold_idx in range(k):
            test_idx = folds[fold_idx]
            train_idx = np.concatenate([folds[j] for j in range(k) if j != fold_idx])

            train_set = [samples[j] for j in train_idx]
            test_set = [samples[j] for j in test_idx]

            train_outputs, _ = run_pipeline(
                task_id_list,
                {
                    EvaluatorISName.SAMPLES: train_set,
                    **inputs,
                },
            )

            x_mean = train_outputs[StatOSName.X_MEAN]
            x_std = train_outputs[StatOSName.X_STANDARD_DEVIATION]
            y_mean = train_outputs[StatOSName.Y_MEAN]
            y_std = train_outputs[StatOSName.Y_STANDARD_DEVIATION]

            model = train_outputs[ModelOSName.MODEL]

            x_test = np.array([x for x, _ in test_set])
            x_test_std = (x_test - x_mean) / x_std
            y_true = np.array([y for _, y in test_set], dtype=float)
            y_pred_std = model.predict(x_test_std)
            y_pred = y_pred_std * y_std + y_mean

            maes.append(mean_absolute_error(y_true, y_pred))
            rmses.append(np.sqrt(mean_squared_error(y_true, y_pred)))
            r2s.append(r2_score(y_true, y_pred))

            denom = np.where(np.abs(y_true) < 1e-8, np.nan, np.abs(y_true))
            mapes.append(float(np.nanmean(np.abs((y_true - y_pred) / denom) * 100.0)))

        # ---- Populate outputs
        self.set_output(EvaluatorOSName.MAE, float(np.mean(maes)))
        self.set_output(EvaluatorOSName.RMSE, float(np.mean(rmses)))
        self.set_output(EvaluatorOSName.R2, float(np.mean(r2s)))
        self.set_output(EvaluatorOSName.MAPE, float(np.mean(mapes)))


# @task(
#     id='SVR-TUNER',
#     kind=EVALUATOR_KIND,
#     description='Evaluates a trained 2D polynomial regression model on test samples.',
#     input_specs={
#         EvaluatorISName.SAMPLES: IOSpec(dtype=LabeledSamples),
#         EvaluatorISName.REGULARIZATION_PARAMETERS: IOSpec(dtype=str),
#         EvaluatorISName.EPSILONS: IOSpec(dtype=str),
#         EvaluatorISName.INDICATOR: IOSpec(dtype=str),
#     },
#     output_specs={
#         EvaluatorOSName.BEST_REGULARIZATION_PARAMETER: IOSpec(dtype=float),
#         EvaluatorOSName.BEST_EPSILON: IOSpec(dtype=float),
#     },
# )
# class SVRTuner(Task):
#     @override
#     def run(self) -> None:
#         c_list = [
#             float(c_str)
#             for c_str in self.get_input(
#                 EvaluatorISName.REGULARIZATION_PARAMETERS
#             ).split(',')
#         ]
#         epsilon_list = [
#             float(e_str)
#             for e_str in self.get_input(EvaluatorISName.EPSILONS).split(',')
#         ]
#
#         samples = self.get_input(EvaluatorISName.SAMPLES)
#
#         indicator = self.get_input(EvaluatorISName.INDICATOR)
#         want_bigger = True if indicator == 'r2' else False
#
#         best_score = -float('inf') if want_bigger else float('inf')
#         best_comb = (0.0, 0.0)
#         total = len(c_list) * len(epsilon_list)
#         i = 1
#         for c in c_list:
#             for epsilon in epsilon_list:
#                 outputs, _ = run_pipeline(
#                     ['CROSS-VALIDATION'],
#                     {
#                         EvaluatorISName.SAMPLES: [(x[:], y) for x, y in samples],
#                         EvaluatorISName.TRAINING_PIPE: 'ST-BASIC MD-IF MD-SVR',
#                         ModelISName.REGULARIZATION_PARAMETER: c,
#                         ModelISName.EPSILON: epsilon,
#                     },
#                 )
#
#                 score = outputs[indicator]
#                 if want_bigger and score > best_score:
#                     best_score = score
#                     best_comb = (c, epsilon)
#                 elif not want_bigger and score < best_score:
#                     best_score = score
#                     best_comb = (c, epsilon)
#
#                 print(
#                     f'[{i}/{total}] Evaluated C={c}, epsilon={epsilon}: '
#                     f'score={score:.3}'
#                 )
#
#                 i += 1
#
#         # ---- Populate outputs
#         self.set_output(EvaluatorOSName.BEST_REGULARIZATION_PARAMETER, best_comb[0])
#         self.set_output(EvaluatorOSName.BEST_EPSILON, best_comb[1])


@task(
    id='SVR-TUNER',
    kind=EVALUATOR_KIND,
    description='Evaluates a trained 2D polynomial regression model on test samples.',
    input_specs={
        EvaluatorISName.SAMPLES: IOSpec(dtype=LabeledSamples),
        EvaluatorISName.INDICATOR: IOSpec(dtype=str),
    },
    output_specs={
        EvaluatorISName.INDICATOR: IOSpec(dtype=str),
        EvaluatorOSName.BEST_REGULARIZATION_PARAMETER: IOSpec(dtype=float),
        EvaluatorOSName.BEST_EPSILON: IOSpec(dtype=float),
        EvaluatorOSName.BEST_SCORE: IOSpec(dtype=float),
    },
)
class SVRTuner(Task):
    @override
    def run(self) -> None:
        samples = self.get_input(EvaluatorISName.SAMPLES)
        indicator = self.get_input(EvaluatorISName.INDICATOR).strip().lower()
        want_bigger = True if indicator == 'r2' else False

        # Random coarse search
        c_range = (0.01, 100.0)
        epsilon_range = (0.001, 1.0)
        rng = Random(42)

        comb_scores: List[Tuple[Tuple[float, float], float]] = []
        total_runs = 100
        for i in range(total_runs):
            c = round(rng.uniform(*c_range), 3)
            epsilon = round(rng.uniform(*epsilon_range), 3)
            score = self._cross_validation(samples, c, epsilon, indicator)

            comb_scores.append(((c, epsilon), score))

            print(
                f'[Coarse:{i + 1}/{total_runs}] Evaluated C={c:.3}, '
                f'epsilon={epsilon:.3}: score={score:.3}'
            )

        comb_scores.sort(key=lambda x: x[1], reverse=want_bigger)

        # Find the 10% best combinations
        top_n = comb_scores[: max(1, total_runs // 10)]

        c_error_bound = 10
        epsilon_error_bound = 0.2
        c_ranges = [
            (max(0.1, comb[0] - c_error_bound), comb[0] + c_error_bound)
            for comb, _ in top_n
        ]
        epsilon_ranges = [
            (max(0.001, comb[1] - epsilon_error_bound), comb[1] + epsilon_error_bound)
            for comb, _ in top_n
        ]

        comb_scores = []
        total_runs = 100
        for i in range(total_runs):
            c_range_idx = rng.randint(0, len(c_ranges) - 1)
            epsilon_range_idx = rng.randint(0, len(epsilon_ranges) - 1)

            c = round(rng.uniform(*c_ranges[c_range_idx]), 3)
            epsilon = round(rng.uniform(*epsilon_ranges[epsilon_range_idx]), 3)
            score = self._cross_validation(samples, c, epsilon, indicator)
            comb_scores.append(((c, epsilon), score))

            print(
                f'[Refine {i + 1}/{total_runs}] Evaluated C={c:.3}, '
                f'epsilon={epsilon:.3}: score={score:.3}'
            )

        comb_scores.sort(key=lambda x: x[1], reverse=want_bigger)

        # Refine search stage 2 (5%)
        top_n = comb_scores[: max(1, total_runs // 20)]
        c_error_bound = 1
        epsilon_error_bound = 0.05
        c_ranges = [
            (max(0.1, comb[0] - c_error_bound), comb[0] + c_error_bound)
            for comb, _ in top_n
        ]
        epsilon_ranges = [
            (max(0.001, comb[1] - epsilon_error_bound), comb[1] + epsilon_error_bound)
            for comb, _ in top_n
        ]

        comb_scores = []
        total_runs = 100
        for i in range(total_runs):
            c_range_idx = rng.randint(0, len(c_ranges) - 1)
            epsilon_range_idx = rng.randint(0, len(epsilon_ranges) - 1)

            c = round(rng.uniform(*c_ranges[c_range_idx]), 3)
            epsilon = round(rng.uniform(*epsilon_ranges[epsilon_range_idx]), 3)
            score = self._cross_validation(samples, c, epsilon, indicator)
            comb_scores.append(((c, epsilon), score))

            print(
                f'[Refine-2 {i + 1}/{total_runs}] Evaluated C={c:.3}, '
                f'epsilon={epsilon:.3}: score={score:.3}'
            )

        comb_scores.sort(key=lambda x: x[1], reverse=want_bigger)

        # ---- Populate outputs
        self.set_output(
            EvaluatorOSName.BEST_REGULARIZATION_PARAMETER, comb_scores[0][0][0]
        )
        self.set_output(EvaluatorOSName.BEST_EPSILON, comb_scores[0][0][1])
        self.set_output(EvaluatorOSName.BEST_SCORE, round(comb_scores[0][1], 3))

    def _cross_validation(self, samples, c, epsilon, indicator) -> float:
        outputs, _ = run_pipeline(
            ['CROSS-VALIDATION'],
            {
                EvaluatorISName.SAMPLES: [(x[:], y) for x, y in samples],
                EvaluatorISName.TRAINING_PIPE: 'ST-BASIC MD-IF MD-SVR',
                ModelISName.REGULARIZATION_PARAMETER: c,
                ModelISName.EPSILON: epsilon,
            },
        )

        return outputs[indicator]
