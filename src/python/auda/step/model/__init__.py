from typing import Any, Type

from auda.utils.pipeline import Step


class ModelBasedStep(Step):
    def verify_model(self, model: Any, expected_model_type: Type) -> None:
        if model is None:
            raise ValueError('Model cannot be None.')

        if not isinstance(model, expected_model_type):
            raise TypeError(
                f'Expected model of type {expected_model_type.__name__}, '
                f'but got {type(model).__name__}.'
            )
