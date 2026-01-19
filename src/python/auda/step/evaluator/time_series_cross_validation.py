from typing import List, override

import numpy as np
from auda.step import create_pipeline_from_pipe
from auda.step.dataset import DatasetBasedStep
from auda.step.evaluator import mae, mape, mse, r2
from auda.step.spec import Dataset, Spec
from auda.utils.pipeline import IOValueMap, Pipeline, step


@step(
    id='EV-TSCV',
    description='Time-series cross validation evaluator (forward-chaining, '
    + 'blocked).',
    input_specs=[
        Spec.ON.optional(Spec.DATASET.name),
        # Reuse existing port name; interpretation differs from EV-CV:
        #   num_k_folds == number of forward validation blocks
        Spec.NUM_K_FOLDS.optional(5),
        Spec.PIPE,
        Spec.SEED.optional(42),
    ],
    output_specs=[Spec.MAE, Spec.RMSE, Spec.R2, Spec.MAPE],
)
class TimeSeriesCrossValidation(DatasetBasedStep):
    @override
    def run(
        self,
        on: str | Dataset,
        num_k_folds: int,
        pipe: str | Pipeline,
        seed: int,
    ) -> IOValueMap:
        X, y = self.get_dataset_from_on(on)
        X = np.asarray(X)
        y = np.asarray(y).reshape(-1)

        n = X.shape[0]
        if n < 3:
            raise ValueError(
                'TimeSeriesCrossValidation requires at least 3 samples.'
            )

        # ---- Sort by time (assumes time is the first feature column)
        order = np.argsort(X[:, 0])
        X = X[order]
        y = y[order]

        # ---- Choose contiguous blocks:
        # Split into (num_k_folds + 1) blocks:
        #   block[0] = initial training block
        #   block[1..] = validation blocks evaluated sequentially
        if num_k_folds < 1 or num_k_folds >= n:
            raise ValueError(
                f'num_k_folds must be >= 1 and < {n}, but got {num_k_folds}.'
            )

        blocks = np.array_split(np.arange(n), num_k_folds + 1)

        # Require at least 2 points to train in the first block (helps many
        # models)
        if len(blocks[0]) < 2:
            raise ValueError(
                'Initial training block is too small. '
                'Reduce num_k_folds or provide more data.'
            )

        maes: List[float] = []
        rmses: List[float] = []
        r2s: List[float] = []
        mapes: List[float] = []

        pipeline = create_pipeline_from_pipe(pipe)

        # Forward-chaining evaluation
        # If blocks = [B0, B1, B2, B3], then folds are:
        #   Fold 0: train on B0, test on B1
        #   Fold 1: train on B0 + B1, test on B2
        #   Fold 2: train on B0 + B1 + B2, test on B3
        for fold_idx in range(num_k_folds):
            train_idx = np.concatenate(blocks[: fold_idx + 1])
            test_idx = blocks[fold_idx + 1]

            # Skip empty test blocks (can happen with extreme splits)
            if test_idx.size == 0:
                continue

            X_train, y_train = X[train_idx], y[train_idx]
            X_test, y_test = X[test_idx], y[test_idx]

            pipeline.reset().run(
                {
                    **self._inputs,
                    Spec.TRAINIING_SET.name: (X_train, y_train),
                    Spec.ON.name: Spec.TRAINIING_SET.name,
                    Spec.SEED.name: seed,
                }
            )

            model = pipeline.get_value(Spec.MODEL.name)

            # Standardize test features using training stats if present.
            # (EV-CV assumes these exist; we keep compatibility.)
            X_mean = pipeline.get_value(Spec.X_MEAN.name)
            X_std = pipeline.get_value(Spec.X_STD.name)

            if X_mean is not None and X_std is not None:
                X_mean = np.asarray(X_mean)
                X_std = np.asarray(X_std)
                X_std_safe = np.where(X_std == 0, 1, X_std)
                X_test_std = (X_test - X_mean) / X_std_safe
            else:
                X_test_std = X_test

            y_pred_std = np.asarray(
                model.predict(X_test_std), dtype=float
            ).reshape(-1)

            # De-standardize predictions if y stats are available
            y_mean = pipeline.get_value(Spec.Y_MEAN.name)
            y_std = pipeline.get_value(Spec.Y_STD.name)
            if y_mean is not None and y_std is not None:
                y_pred = y_pred_std * float(np.asarray(y_std)) + float(
                    np.asarray(y_mean)
                )
            else:
                y_pred = y_pred_std

            maes.append(mae(y_test, y_pred))
            rmses.append(mse(y_test, y_pred) ** 0.5)
            r2s.append(r2(y_test, y_pred))
            mapes.append(mape(y_test, y_pred))

        if len(maes) == 0:
            raise ValueError(
                'No folds were evaluated (all test blocks were empty). '
                'Reduce num_k_folds or provide more data.'
            )

        return {
            Spec.MAE.name: float(np.mean(maes)),
            Spec.RMSE.name: float(np.mean(rmses)),
            Spec.R2.name: float(np.mean(r2s)),
            Spec.MAPE.name: float(np.mean(mapes)),
        }
