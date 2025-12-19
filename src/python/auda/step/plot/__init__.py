from typing import Callable, List

from auda.step.spec import Spec
from auda.utils.pipeline import Step


class PlotStep(Step):
    def check_feature_dimension(
        self,
        expected_dimension: int | None = None,
        checker: Callable[[int], bool] | None = None,
    ) -> None:
        feature_dimension = len(self.get_input(Spec.FEATURE_NAMES.name))

        if expected_dimension and feature_dimension != expected_dimension:
            raise ValueError(
                f'Expected feature dimension {expected_dimension}, but got '
                f'{feature_dimension} for step {self.spec.id}'
            )

        if checker and not checker(feature_dimension):
            raise ValueError(
                f'Invalid feature dimension: {feature_dimension} for step '
                f'{self.spec.id}'
            )

    def combine_names_units(
        self, names: List[str], units: List[str]
    ) -> List[str]:
        return [
            f'{name} ({unit})' if unit else name
            for name, unit in zip(names, units)
        ]
