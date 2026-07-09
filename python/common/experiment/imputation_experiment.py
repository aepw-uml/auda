from typing import Any, Type, override

from common.experiment.regression_experiment import RegressionExperiment
from common.metrics import RegressionMetrics
from common.metrics.regression_metrics import RegressionMetricName
from sklearn.base import np
from step.evaluator.masked_value_validation import masked_value_validation
from step.evaluator.one_standard_error import one_standard_error
from step.model.model import SupervisedLearningModel
from step.model.standardize_regressor import StandardizedRegressor
from step.tuner.grid_search import grid_search
from step.tuner.random_search import random_search
from step.tuner.types import (
    Hyperparameters,
    SamplingScale,
    SearchSpace,
    Trial,
)


class ImputationExperiment(RegressionExperiment):
    @override
    def __init__(
        self,
        name: str,
        description: str,
        regressor_cls: Type[SupervisedLearningModel],
        train_size: float = 0.9,
        seed: int = 471,
    ) -> None:
        super().__init__(name, description, train_size, seed)
        self.regressor_cls = regressor_cls
        self.context['split_shuffle'] = True

    @override
    def split(self) -> None:
        """Splits imputation data while keeping boundary samples in train.

        Imputation models need observed samples on both ends of the
        sequence, so the first and last original samples are always assigned to
        the training split. Test samples are selected only from interior
        positions.

        Raises:
            ValueError: If the dataset has not been provided via ``setup()`` or
                if evaluation is enabled with fewer than three samples.
        """

        if self.X is None or self.y is None:
            raise ValueError('Data not set up. Call setup() first.')

        enable_evaluation = self.get_context_bool('enable_evaluation', True)
        if not enable_evaluation:
            return super().split()

        super().split()
        n_samples = self.X.shape[0]
        if n_samples < 3:
            raise ValueError(
                'Imputation experiments require at least 3 samples when '
                'evaluation is enabled.'
            )

        boundary_indices = np.array([0, n_samples - 1], dtype=int)
        interior_indices = np.arange(1, n_samples - 1)

        if self.get_context_bool('split_shuffle', False):
            np.random.seed(self.seed)
            np.random.shuffle(interior_indices)

        test_size = 1 - self.train_size
        num_test_samples = max(2, int(n_samples * test_size))
        num_test_samples = min(num_test_samples, interior_indices.shape[0])
        test_indices = interior_indices[:num_test_samples]
        train_indices = np.concatenate(
            (boundary_indices, interior_indices[num_test_samples:])
        )

        self.X_train = self.X[train_indices]
        self.y_train = self.y[train_indices]
        self.X_test = self.X[test_indices]
        self.y_test = self.y[test_indices]

        m_train = self.X_train.shape[0]
        m_test = self.X_test.shape[0]
        self.log(
            f'Split data into {m_train} training samples '
            f'and {m_test} test samples, reserving boundary samples for '
            'training.'
        )

    @override
    def train(self) -> None:
        super().train()

        self.model = StandardizedRegressor(
            regressor_cls=self.regressor_cls,
            hyperparameters=self.hyperparameters,
            regressor_kwargs=self.context,
            use_x_scaler=self.get_context_bool('use_scaler', True),
            use_y_scaler=self.get_context_bool('use_target_scaler', True),
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
        metric: RegressionMetricName = tuning_parameters.get('metric', 'wape')
        num_iterations: int = tuning_parameters.get('num_iterations', 128)
        num_points_per_interval: int = tuning_parameters.get(
            'num_points_per_interval', 12
        )
        complexity_key = tuning_parameters.get('complexity_key')
        validation_rate: float = tuning_parameters.get('validation_rate', 0.2)
        num_masks: int = tuning_parameters.get('num_masks', 5)

        if complexity_key is None:
            raise ValueError(
                'tuning_parameters must include "complexity_key" function.'
            )

        def evaluate_mask(
            X_train_mask: np.ndarray,
            y_train_mask: np.ndarray,
            X_val_mask: np.ndarray,
            y_val_mask: np.ndarray,
        ) -> RegressionMetrics:
            X_train, y_train = self.X_train, self.y_train
            X_test, y_test = self.X_test, self.y_test
            enable_logging = self.get_context_bool('enable_logging', True)

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
            trials: list[Trial] = grid_search(
                hyperparameter_names=hyperparameter_names,
                search_space=search_space,
                evaluate_hyperparameters=evaluate_hyperparameters,
                sampling_scales=sampling_scales,
                num_points_per_interval=num_points_per_interval,
                logger=self.logger,
            )
        elif search_type == 'random':
            trials = random_search(
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
        for _, metrics_list in trials:
            scores_list.append(
                [metrics.get_value_by_name(metric) for metrics in metrics_list]
            )

        trial_scores: list[tuple[Hyperparameters, list[float]]] = [
            (trial[0], scores) for trial, scores in zip(trials, scores_list)
        ]
        best_hyperparameters = one_standard_error(
            trial_scores,
            complexity_key,
            prefer_lower=metric != 'r2',
        )

        self.context['trials'] = trials
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
