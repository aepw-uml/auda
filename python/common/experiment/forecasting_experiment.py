from typing import Any, Type, override

from common.experiment.regression_experiment import RegressionExperiment
from common.metrics import RegressionMetrics
from common.metrics.regression_metrics import RegressionMetricName
from sklearn.base import np
from step.evaluator.one_standard_error import one_standard_error
from step.evaluator.time_series_cross_validation import (
    time_series_cross_validation,
)
from step.model.model import SupervisedLearningModel
from step.model.standardize_regressor import StandardizedRegressor
from step.tuner.grid_search import grid_search
from step.tuner.random_search import random_search
from step.tuner.types import (
    Configuration,
    Hyperparameters,
    SamplingScale,
    SearchSpace,
)


class ForecastingExperiment(RegressionExperiment):
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
        search_type = self.context.get('tune_search_type', 'random')
        hyperparameter_names: list[str] = tuning_parameters.get(
            'hyperparameter_names', []
        )
        search_space: SearchSpace = tuning_parameters.get('search_space', [])
        sampling_scales: list[SamplingScale] = tuning_parameters.get(
            'sampling_scales', []
        )
        metric: RegressionMetricName = tuning_parameters.get('metric', 'wape')
        num_iterations: int = tuning_parameters.get('num_iterations', 100)
        num_points_per_interval: int = tuning_parameters.get(
            'num_points_per_interval', 16
        )
        complexity_key = tuning_parameters.get('complexity_key')

        if complexity_key is None:
            raise ValueError(
                'tuning_parameters must include "complexity_key" function.'
            )

        def evaluate_fold(
            X_train_fold: np.ndarray,
            y_train_fold: np.ndarray,
            X_val_fold: np.ndarray,
            y_val_fold: np.ndarray,
        ) -> RegressionMetrics:
            X_train, y_train = self.X_train, self.y_train
            X_test, y_test = self.X_test, self.y_test
            enable_logging = self.context.get('enable_logging', True)

            self.X_train, self.y_train = X_train_fold, y_train_fold
            self.X_test, self.y_test = X_val_fold, y_val_fold
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
            all_regression_metrics = time_series_cross_validation(
                X_train, y_train, evaluate_fold
            )

            return all_regression_metrics

        self.timer_start('tuning')
        if search_type == 'grid':
            configurations: list[Configuration] = grid_search(
                hyperparameter_names=hyperparameter_names,
                search_space=search_space,
                evaluate_hyperparameters=evaluate_hyperparameters,
                sampling_scales=sampling_scales,
                num_points_per_interval=num_points_per_interval,
                logger=self.logger,
            )
        elif search_type == 'random':
            configurations: list[Configuration] = random_search(
                hyperparameter_names=hyperparameter_names,
                search_space=search_space,
                evaluate_hyperparameters=evaluate_hyperparameters,
                sampling_scales=sampling_scales,
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

        scores_list: list[list[float]] = []
        for _, metrics_list in configurations:
            scores_list.append(
                [metrics.get_value_by_name(metric) for metrics in metrics_list]
            )

        configuration_scores: list[tuple[Hyperparameters, list[float]]] = [
            (configuration[0], scores)
            for configuration, scores in zip(configurations, scores_list)
        ]
        best_hyperparameters = one_standard_error(
            configuration_scores,
            complexity_key,
            prefer_lower=metric != 'r2',
        )

        self.context['configuration'] = configurations
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
