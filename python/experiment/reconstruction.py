from typing import Any, Type, override

import numpy as np
from common.metrics import (
    RegressionMetricName,
    RegressionMetrics,
    average_regression_metrics,
)
from experiment.experiment import RegressionExperiment
from sklearn.base import RegressorMixin
from sklearn.ensemble import IsolationForest
from step.evaluator.masked_value_validation import masked_value_validation
from step.model.model import SupervisedLearningModel
from step.model.standardize_regressor import StandardizedRegressor
from step.tuner.multistage_random_search import multistage_random_search
from step.tuner.random_search import Interval, SamplingScale, SearchSpace


class ReconstructionExperiment(RegressionExperiment):
    @override
    def __init__(
        self,
        name: str,
        description: str,
        regressor: Type[SupervisedLearningModel | RegressorMixin],
        seed: int = 42,
        train_rate: float = 0.8,
    ) -> None:
        super().__init__(name, description, seed, train_rate)
        self.regressor = regressor

    @override
    def train(self) -> None:
        super().train()

        training_features, training_targets = self.get_training_set()
        use_isolation_forest: bool = self.context.get(
            'use_isolation_forest', False
        )
        if use_isolation_forest:
            contamination = self.context.get('contamination', 'auto')
            isolation_forest = IsolationForest(
                contamination=contamination,
                random_state=self.seed,
            )
            inlier_mask = isolation_forest.fit_predict(self.X_train) == 1
            self.context['inlier_mask'] = inlier_mask

            inlier_training_features: np.ndarray = training_features[
                inlier_mask
            ]
            inlier_training_targets: np.ndarray = training_targets[inlier_mask]

        self.pipeline = StandardizedRegressor(
            regressor_cls=self.regressor,
            regressor_kwargs=self.context,
            use_x_scaler=self.context.get('use_scaler', True),
            use_y_scaler=self.context.get('use_target_scaler', True),
        )

        if use_isolation_forest:
            self.pipeline.fit(inlier_training_features, inlier_training_targets)
        else:
            self.pipeline.fit(training_features, training_targets)

        self.parameters = self.pipeline.regressor_.parameters

    @override
    def tune(self) -> None:
        tuning_parameters: dict[str, Any] | None = self.context.get(
            'tuning_parameters'
        )
        if tuning_parameters is None:
            return self.logger.info(
                'Skipping tuning since no tuning parameters were provided.'
            )

        self.logger.info('Starting hyperparameter tuning...')
        hyperparameter_names: list[str] = tuning_parameters.get(
            'hyperparameter_names', []
        )
        search_space: SearchSpace = tuning_parameters.get('search_space', [])
        elite_fractions: list[float] = tuning_parameters.get(
            'elite_fractions', []
        )
        refinement_width_rates: list[list[float]] = tuning_parameters.get(
            'refinement_width_rates', []
        )
        hyperparameter_domains: list[Interval] | None = tuning_parameters.get(
            'hyperparameter_domains', None
        )
        sampling_scales: list[SamplingScale] = tuning_parameters.get(
            'sampling_scales', []
        )
        metric: RegressionMetricName = tuning_parameters.get('metric', 'mape')
        expect_higher: bool = tuning_parameters.get('expect_higher', 'auto')
        num_iterations: list[int] = tuning_parameters.get(
            'num_iterations', [50, 10, 5]
        )
        num_masks: int = tuning_parameters.get('num_masks', 5)

        training_features, training_targets = self.get_training_set()

        def evaluate_fold(
            fold_training_features: np.ndarray,
            fold_training_targets: np.ndarray,
            fold_validation_features: np.ndarray,
            fold_validation_targets: np.ndarray,
        ) -> RegressionMetrics:
            original_training_features = self.X_train
            original_training_targets = self.y_train
            original_test_features = self.X_test
            original_test_targets = self.y_test

            self.X_train, self.y_train = (
                fold_training_features,
                fold_training_targets,
            )
            self.X_test, self.y_test = (
                fold_validation_features,
                fold_validation_targets,
            )
            self.context['enable_logging'] = False

            self.train()
            self.evaluate()
            metrics = self.get_metrics()

            self.X_train, self.y_train = (
                original_training_features,
                original_training_targets,
            )
            self.X_test, self.y_test = (
                original_test_features,
                original_test_targets,
            )
            self.context['enable_logging'] = True

            return metrics

        def evaluate_hyperparameters(
            hyperparameters: list[float],
        ) -> RegressionMetrics:
            self.set_hyperparameters(
                replace=True,
                **{
                    name: value
                    for name, value in zip(
                        hyperparameter_names, hyperparameters
                    )
                },
            )

            metrics_list = masked_value_validation(
                training_features,
                training_targets,
                evaluate=evaluate_fold,
                validation_rate=0.1,
                num_masks=num_masks,
                seed=self.seed,
            )

            return average_regression_metrics(metrics_list)

        _, best_hyperparameters = multistage_random_search(
            hyperparameter_names,
            search_space,
            elite_fractions,
            refinement_width_rates,
            evaluate_hyperparameters,
            sampling_scales,
            hyperparameter_domains,
            metric,
            expect_higher,
            num_iterations,
            self.seed,
            self.logger,
        )

        self.set_hyperparameters(
            replace=True,
            **{
                name: value
                for name, value in zip(
                    hyperparameter_names, best_hyperparameters
                )
            },
        )
