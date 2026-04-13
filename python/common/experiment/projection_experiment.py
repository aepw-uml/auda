from typing import Any, Type, override

from common.experiment.regression_experiment import RegressionExperiment
from sklearn.base import RegressorMixin
from step.model.model import SupervisedLearningModel
from step.model.standardize_regressor import StandardizedRegressor


class ProjectionExperiment(RegressionExperiment):
    @override
    def __init__(
        self,
        name: str,
        description: str,
        regressor: Type[SupervisedLearningModel | RegressorMixin],
        train_size: float = 0.9,
        seed: int = 417,
    ) -> None:
        super().__init__(name, description, train_size, seed)
        self.regressor = regressor
        self.context['split_shuffle'] = False

    @override
    def train(self) -> None:
        super().train()

        self.pipeline = StandardizedRegressor(
            regressor_cls=self.regressor,
            regressor_kwargs=self.context,
            use_x_scaler=self.context.get('use_scaler', True),
            use_y_scaler=self.context.get('use_target_scaler', True),
        )

        self.parameters = self.pipeline.regressor_.parameters

    @override
    def tune(self) -> None:
        # TODO: Change this
        tuning_parameters: dict[str, Any] | None = self.context.get(
            'tuning_parameters'
        )
        if tuning_parameters is None:
            return self.logger.info(
                'Skipping tuning since no tuning parameters were provided.'
            )
