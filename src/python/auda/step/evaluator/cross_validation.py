from typing import List, override

import numpy as np
from auda.step import create_pipeline_from_pipe
from auda.step.dataset import DatasetBasedStep
from auda.step.evaluator import mae, mape, mse, r2
from auda.step.spec import Dataset, Spec
from auda.utils.pipeline import (
    IOValueMap,
    Pipeline,
    step,
)


@step(
    id='EV-CV',
    description='Cross Validation Evaluator.',
    input_specs=[
        Spec.ON.optional(Spec.DATASET.name),
        Spec.SEED.optional(42),
        Spec.NUM_K_FOLDS.optional(5),
        Spec.PIPE,
    ],
    output_specs=[Spec.MAE, Spec.RMSE, Spec.R2, Spec.MAPE],
)
class CrossValidationEvaluator(DatasetBasedStep):
    @override
    def run(
        self,
        on: str | Dataset,
        seed: int,
        num_k_folds: int,
        pipe: str | Pipeline,
    ) -> IOValueMap:
        X, y = self.get_dataset_from_on(on)
        n = X.shape[0]

        # ---- Check num_k_folds
        if num_k_folds <= 1 or num_k_folds > (n // 2):
            raise ValueError(
                f'num_k_folds must be > 1 and <= {n // 2}, but '
                f'got {num_k_folds}.'
            )

        # ---- Shuffle dataset
        rng = np.random.default_rng(seed)
        perm = rng.permutation(n)
        X = X[perm]
        y = y[perm]

        # ---- Cross Validation
        # Store evaluation metrics for each fold
        maes: List[float] = []
        rmses: List[float] = []
        r2s: List[float] = []
        mapes: List[float] = []

        # Create a pipeline
        pipeline = create_pipeline_from_pipe(pipe)

        folds = np.array_split(np.arange(n), num_k_folds)
        for fold_idx in range(num_k_folds):
            test_idx = folds[fold_idx]
            X_test = X[test_idx]
            y_test = y[test_idx]
            X_train = np.concatenate(
                [X[folds[j]] for j in range(num_k_folds) if j != fold_idx]
            )
            y_train = np.concatenate(
                [y[folds[j]] for j in range(num_k_folds) if j != fold_idx]
            )

            pipeline.reset().run(
                {
                    **self._inputs,
                    Spec.TRAINIING_SET.name: (X_train, y_train),
                    Spec.ON.name: Spec.TRAINIING_SET.name,
                    Spec.SEED.name: seed,
                }
            )
            model = pipeline.get_value(Spec.MODEL.name)

            # Make predictions
            X_mean = pipeline.get_value(Spec.X_MEAN.name)
            X_std = pipeline.get_value(Spec.X_STD.name)
            X_test_std = (X_test - X_mean[0]) / X_std[0]

            model.predict(X_test_std)
            y_mean = pipeline.get_value(Spec.Y_MEAN.name)
            y_std = pipeline.get_value(Spec.Y_STD.name)
            y_pred_std = model.predict(X_test_std)
            y_pred = y_pred_std * y_std + y_mean

            # Calculate evaluation metrics
            maes.append(mae(y_test, y_pred))
            rmses.append(mse(y_test, y_pred) ** 0.5)
            r2s.append(r2(y_test, y_pred))
            mapes.append(mape(y_test, y_pred))

        return {
            Spec.MAE.name: np.mean(maes),
            Spec.RMSE.name: np.mean(rmses),
            Spec.R2.name: np.mean(r2s),
            Spec.MAPE.name: np.mean(mapes),
        }
