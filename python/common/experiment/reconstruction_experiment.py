from typing import Any, Type, override

from common.experiment.regression_experiment import RegressionExperiment
from common.metrics import RegressionMetrics
from common.metrics.regression_metrics import RegressionMetricName
from sklearn.base import np
from step.evaluator.masked_value_validation import masked_value_validation
from step.model.model import SupervisedLearningModel
from step.model.standardize_regressor import StandardizedRegressor
from step.tuner.grid_search import grid_search
from step.tuner.random_search import random_search
from step.tuner.types import (
    HyperparameterScore,
    SamplingScale,
    SearchSpace,
)


class ReconstructionExperiment(RegressionExperiment):
    @override
    def __init__(
        self,
        name: str,
        description: str,
        regressor_cls: Type[SupervisedLearningModel],
        train_size: float = 0.9,
        seed: int = 417,
    ) -> None:
        super().__init__(name, description, train_size, seed)
        self.regressor_cls = regressor_cls
        self.context['split_shuffle'] = False

    @override
    def train(self) -> None:
        super().train()

        self.model = StandardizedRegressor(
            regressor_cls=self.regressor_cls,
            hyperparameters=self.hyperparameters,
            regressor_kwargs=self.context,
            use_x_scaler=self.context.get('use_scaler', True),
            use_y_scaler=self.context.get('use_target_scaler', True),
        )

        self.model.fit(self.X_train, self.y_train)
        self.parameters = self.model.regressor_.parameters

        self.logger.info('Model trained.')

    @override
    def tune(self) -> None:
        tuning_parameters: dict[str, Any] | None = self.context.get(
            'tuning_parameters'
        )
        if tuning_parameters is None:
            return self.logger.info(
                'Skipping tuning since no tuning parameters were provided.'
            )

        self.logger.info('Tuning model with provided parameters...')
        search_type = tuning_parameters.get(
            'search_type',
            self.context.get('tune_search_type', 'random'),
        )
        hyperparameter_names: list[str] = tuning_parameters.get(
            'hyperparameter_names', []
        )
        search_space: SearchSpace = tuning_parameters.get('search_space', [])
        sampling_scales: list[SamplingScale] = tuning_parameters.get(
            'sampling_scales', []
        )
        metric: RegressionMetricName = tuning_parameters.get('metric', 'mape')
        num_iterations: int = tuning_parameters.get('num_iterations', 500)
        num_points_per_interval: int = tuning_parameters.get(
            'num_points_per_interval', 5
        )
        validation_rate: float = tuning_parameters.get('validation_rate', 0.2)
        num_masks: int = tuning_parameters.get('num_masks', 5)

        def evaluate_mask(
            X_train_mask: np.ndarray,
            y_train_mask: np.ndarray,
            X_val_mask: np.ndarray,
            y_val_mask: np.ndarray,
        ) -> RegressionMetrics:
            X_train, y_train = self.X_train, self.y_train
            X_test, y_test = self.X_test, self.y_test
            enable_logging = self.context.get('enable_logging', True)

            self.X_train, self.y_train = X_train_mask, y_train_mask
            self.X_test, self.y_test = X_val_mask, y_val_mask
            self.context['enable_logging'] = False

            try:
                self.train()
                self.evaluate()
                return self.get_metrics()
            finally:
                self.X_train, self.y_train = X_train, y_train
                self.X_test, self.y_test = X_test, y_test
                self.context['enable_logging'] = enable_logging

        X_train, y_train = self.get_training_set()

        def evaluate_hyperparameters(
            hyperparameters: list[float],
        ) -> list[RegressionMetrics]:
            self.hyperparameters = {
                **self.hyperparameters,
                **{
                    name: value
                    for name, value in zip(
                        hyperparameter_names, hyperparameters
                    )
                },
            }
            return masked_value_validation(
                X=X_train,
                y=y_train,
                evaluate=evaluate_mask,
                validation_rate=validation_rate,
                num_masks=num_masks,
                seed=self.seed,
            )

        self.timer_start('tuning')
        if search_type == 'grid':
            hyperparameter_scores: list[HyperparameterScore] = grid_search(
                hyperparameter_names=hyperparameter_names,
                search_space=search_space,
                evaluate_hyperparameters=evaluate_hyperparameters,
                sampling_scales=sampling_scales,
                metric=metric,
                num_points_per_interval=num_points_per_interval,
                logger=self.logger,
            )
        elif search_type == 'random':
            hyperparameter_scores = random_search(
                hyperparameter_names=hyperparameter_names,
                search_space=search_space,
                evaluate_hyperparameters=evaluate_hyperparameters,
                sampling_scales=sampling_scales,
                metric=metric,
                num_iterations=num_iterations,
                seed=self.seed,
                logger=self.logger,
            )
        else:
            raise ValueError(
                f'Unsupported search_type "{search_type}". '
                'Expected "grid" or "random".'
            )
        self.timer_stop('tuning')

        sorted_hyperparameter_scores = sorted(
            hyperparameter_scores, key=lambda x: x[0]
        )
        index = int(len(hyperparameter_scores) * 0.01)
        best_hyperparameters = sorted_hyperparameter_scores[index][1]

        self.context['hyperparameter_scores'] = hyperparameter_scores
        self.hyperparameters = {
            **self.hyperparameters,
            **{
                name: value
                for name, value in zip(
                    hyperparameter_names, best_hyperparameters
                )
            },
        }

        self.logger.info(f'Best hyperparameters: {self.hyperparameters}')
