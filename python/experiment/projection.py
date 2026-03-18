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
from step.evaluator.time_series_cross_validation import (
    time_series_cross_validation,
)
from step.model.model import SupervisedLearningModel
from step.model.standardize_regressor import StandardizedRegressor
from step.tuner.multistage_random_search import multistage_random_search
from step.tuner.random_search import Interval, SamplingScale, SearchSpace


class ProjectionExperiment(RegressionExperiment):
    @override
    def __init__(
        self,
        name: str,
        description: str,
        regressor: Type[SupervisedLearningModel | RegressorMixin],
        seed: int = 42,
        train_rate: float = 0.9,
    ) -> None:
        super().__init__(name, description, seed, train_rate)
        self.regressor = regressor
        self.context['split_shuffle'] = False

    @override
    def train(self) -> None:
        super().train()

        X_train, y_train = self.get_training_set()
        use_isolation_forest: bool = self.context.get(
            'use_isolation_forest', False
        )
        if use_isolation_forest:
            contamination = self.context.get('contamination', 'auto')
            iso = IsolationForest(
                contamination=contamination,
                random_state=self.seed,
            )
            inlier_mask = iso.fit_predict(self.X_train) == 1
            self.context['inlier_mask'] = inlier_mask

            X_train_inliers: np.ndarray = X_train[inlier_mask]
            y_train_inliers: np.ndarray = y_train[inlier_mask]

        self.pipeline = StandardizedRegressor(
            regressor_cls=self.regressor,
            regressor_kwargs=self.context,
            use_x_scaler=self.context.get('use_scaler', True),
            use_y_scaler=self.context.get('use_target_scaler', True),
        )

        if use_isolation_forest:
            self.pipeline.fit(X_train_inliers, y_train_inliers)
        else:
            self.pipeline.fit(X_train, self.y_train)

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
            'num_iterations', [200, 20, 10]
        )

        X_train, y_train = self.get_training_set()

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
            all_regression_metrics = time_series_cross_validation(
                X_train, y_train, evaluate_fold
            )

            return average_regression_metrics(all_regression_metrics)

        self.timer_start('tuning')
        hyperparameter_scores_lists, best_hyperparameters = (
            multistage_random_search(
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
        )
        self.timer_stop('tuning')

        self.set_hyperparameters(
            replace=True,
            **{
                name: value
                for name, value in zip(
                    hyperparameter_names, best_hyperparameters
                )
            },
        )

        self.context['hyperparameter_scores_lists'] = (
            hyperparameter_scores_lists
        )
