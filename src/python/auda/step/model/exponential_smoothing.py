from typing import override

import numpy as np
from auda.step.dataset import DatasetBasedStep
from auda.step.spec import Dataset, Spec
from auda.utils.pipeline import IOValueMap, step


class ETSAdapter:
    def __init__(self, fitted_model, train_len: int):
        self._model = fitted_model
        self._train_len = train_len

    def predict(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X)
        n = X.shape[0]

        y_hat = self._model.forecast(steps=n)

        return np.asarray(y_hat, dtype=float)


@step(
    id='MD-ETS',
    description='Exponential smoothing baseline (Holt trend).',
    input_specs=[Spec.ON.optional(Spec.DATASET.name)],
    output_specs=[Spec.MODEL],
)
class ExponentialSmoothingBaseline(DatasetBasedStep):
    @override
    def run(self, on: str | Dataset) -> IOValueMap:
        import numpy as np
        from statsmodels.tsa.holtwinters import ExponentialSmoothing

        X, y = self.get_dataset_from_on(on)

        # Ensure temporal ordering
        order = np.argsort(X[:, 0])
        y_sorted = y[order]

        model = ExponentialSmoothing(
            y_sorted,
            trend='add',
            seasonal=None,
            initialization_method='estimated',
        ).fit()

        model = ETSAdapter(model, train_len=len(y_sorted))

        return {Spec.MODEL.name: model}
