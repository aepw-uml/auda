from typing import Any, Type, override

from common.experiment.regression_experiment import RegressionExperiment
from common.metrics import RegressionMetrics, average_regression_metrics
from common.metrics.regression_metrics import RegressionMetricName
from sklearn.base import np
from step.evaluator.time_series_cross_validation import (
    time_series_cross_validation,
)
from step.model.model import SupervisedLearningModel
from step.model.standardize_regressor import StandardizedRegressor
from step.tuner.random_search import (
    HyperparameterScore,
    SamplingScale,
    SearchSpace,
    random_search,
)


class ProjectionExperiment(RegressionExperiment):
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
        search_type = tuning_parameters.get('search_type', 'random')

        if search_type == 'grid':
            pass
        elif search_type == 'random':
            hyperparameter_names: list[str] = tuning_parameters.get(
                'hyperparameter_names', []
            )
            search_space: SearchSpace = tuning_parameters.get(
                'search_space', []
            )
            sampling_scales: list[SamplingScale] = tuning_parameters.get(
                'sampling_scales', []
            )
            metric: RegressionMetricName = tuning_parameters.get(
                'metric', 'mape'
            )
            num_iterations: int = tuning_parameters.get('num_iterations', 200)

            def evaluate_fold(
                X_train_fold: np.ndarray,
                y_train_fold: np.ndarray,
                X_val_fold: np.ndarray,
                y_val_fold: np.ndarray,
            ) -> RegressionMetrics:
                X_train, y_train = self.X_train, self.y_train
                X_test, y_test = self.X_test, self.y_test

                self.X_train, self.y_train = X_train_fold, y_train_fold
                self.X_test, self.y_test = X_val_fold, y_val_fold
                self.context['enable_logging'] = False

                self.train()
                self.evaluate()
                metrics = self.get_metrics()

                self.X_train, self.y_train = X_train, y_train
                self.X_test, self.y_test = X_test, y_test
                self.context['enable_logging'] = True

                return metrics

            X_train, y_train = self.get_training_set()

            def evaluate_hyperparameters(
                hyperparameters: list[float],
            ) -> RegressionMetrics:
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

                return average_regression_metrics(all_regression_metrics)

            self.timer_start('tuning')
            hyperparameter_scores: list[HyperparameterScore] = random_search(
                hyperparameter_names=hyperparameter_names,
                search_space=search_space,
                evaluate_hyperparameters=evaluate_hyperparameters,
                sampling_scales=sampling_scales,
                metric=metric,
                num_iterations=num_iterations,
                seed=self.seed,
                logger=self.logger,
            )
            self.timer_stop('tuning')

            # TODO: Pick the good but not the best hyperparameters to avoid
            # overfitting to the validation set.
            sorted_hyperparameter_scores = sorted(
                hyperparameter_scores, key=lambda x: x[0]
            )
            best_hyperparameters = sorted_hyperparameter_scores[0][1]

            self.context['hyperparameter_scores'] = hyperparameter_scores
            self.hyperparameters = {
                name: value
                for name, value in zip(
                    hyperparameter_names, best_hyperparameters
                )
            }

        self.logger.info(f'Best hyperparameters: {self.hyperparameters}')
