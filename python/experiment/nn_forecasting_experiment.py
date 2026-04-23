from collections.abc import Callable
from copy import deepcopy
from typing import Literal, cast, override

import numpy as np
import torch
from common.dataset import Dataset, DatasetSchema
from common.experiment.regression_experiment import RegressionExperiment
from common.task import Task
from sklearn.preprocessing import StandardScaler
from torch import nn

TargetTransformName = Literal['none', 'log1p']


def _resolve_target_transform(value: object) -> TargetTransformName:
    """Validates and returns the configured target transform name.

    Args:
        value: Raw transform name from experiment context.

    Returns:
        The validated target transform name.

    Raises:
        ValueError: If the provided transform name is unsupported.
    """

    if value is None:
        return 'none'

    transform = str(value).strip().lower()
    if transform in ('', 'none'):
        return 'none'
    if transform == 'log1p':
        return 'log1p'

    raise ValueError(f'Unsupported target transform: {value}')


def _transform_targets(
    y: np.ndarray,
    target_transform: TargetTransformName,
) -> np.ndarray:
    """Transforms target values for model training.

    Args:
        y: Target array on the original scale.
        target_transform: Configured transform name.

    Returns:
        Target array in the model training space.

    Raises:
        ValueError: If ``log1p`` is requested for negative targets.
    """

    if target_transform == 'none':
        return y

    if np.any(y < 0):
        raise ValueError('log1p target transform requires non-negative y.')

    return np.log1p(y)


def _inverse_transform_targets(
    y: np.ndarray,
    target_transform: TargetTransformName,
) -> np.ndarray:
    """Restores target values from model space to the original scale.

    Args:
        y: Target array in the model training space.
        target_transform: Configured transform name.

    Returns:
        Target array on the original scale.
    """

    if target_transform == 'none':
        return y

    return np.expm1(y)


def _split_training_validation(
    X: np.ndarray,
    y: np.ndarray,
    validation_fraction: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Splits arrays into subtraining and validation partitions.

    Args:
        X: Feature matrix.
        y: Target vector.
        validation_fraction: Fraction of samples reserved for validation.
        seed: Random seed for the shuffle.

    Returns:
        Subtraining features, subtraining targets, validation features, and
        validation targets.
    """

    n_samples = X.shape[0]
    indices = np.arange(n_samples)
    rng = np.random.default_rng(seed)
    rng.shuffle(indices)

    num_validation = max(1, int(n_samples * validation_fraction))
    num_validation = min(num_validation, n_samples - 1)
    train_end = n_samples - num_validation

    train_indices = indices[:train_end]
    validation_indices = indices[train_end:]

    return (
        X[train_indices],
        y[train_indices],
        X[validation_indices],
        y[validation_indices],
    )


def _create_dataloader(
    X: np.ndarray,
    y: np.ndarray,
    batch_size: int,
    seed: int,
    shuffle: bool,
) -> torch.utils.data.DataLoader:
    """Builds a PyTorch dataloader from NumPy arrays.

    Args:
        X: Feature matrix.
        y: Target vector.
        batch_size: Number of samples per batch.
        seed: Random seed for dataloader shuffling.
        shuffle: Whether to shuffle the dataset before each epoch.

    Returns:
        Dataloader over the provided arrays.
    """

    dataset = torch.utils.data.TensorDataset(
        torch.tensor(X, dtype=torch.float32),
        torch.tensor(y, dtype=torch.float32),
    )
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        generator=torch.Generator().manual_seed(seed),
    )


def _fit_model(
    model: '_ForecastingMLP',
    dataloader: torch.utils.data.DataLoader,
    learning_rate: float,
    weight_decay: float,
    num_epochs: int,
    validation_data: tuple[np.ndarray, np.ndarray] | None = None,
    early_stopping_patience: int | None = None,
    early_stopping_min_delta: float = 0.0,
    logger: Callable[[str], None] | None = None,
) -> tuple[int, float | None]:
    """Fits the model and optionally applies validation-based early stopping.

    Args:
        model: Forecasting network to optimize.
        dataloader: Training batches.
        learning_rate: Optimizer learning rate.
        weight_decay: Optimizer weight decay.
        num_epochs: Number of passes through the training data.
        validation_data: Optional validation features and targets in scaled
            space.
        early_stopping_patience: Number of epochs without meaningful
            validation improvement allowed before stopping early.
        early_stopping_min_delta: Minimum validation loss improvement required
            to reset the patience counter.
        logger: Optional callback for progress logging.

    Returns:
        Number of epochs trained and the best validation loss when validation
        data is provided.
    """

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
    )
    loss_fn = nn.MSELoss()
    validation_tensors: tuple[torch.Tensor, torch.Tensor] | None = None
    if validation_data is not None:
        X_validation, y_validation = validation_data
        validation_tensors = (
            torch.tensor(X_validation, dtype=torch.float32),
            torch.tensor(y_validation, dtype=torch.float32),
        )

    best_epoch: int | None = None
    best_state_dict: dict[str, torch.Tensor] | None = None
    best_validation_loss: float | None = None
    epochs_without_improvement = 0

    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0.0
        num_batches = 0
        for X_batch, y_batch in dataloader:
            optimizer.zero_grad()
            predictions = model(X_batch)
            loss = loss_fn(predictions, y_batch)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            num_batches += 1

        average_training_loss = epoch_loss / num_batches
        validation_loss: float | None = None
        if validation_tensors is not None:
            X_validation_tensor, y_validation_tensor = validation_tensors
            model.eval()
            with torch.inference_mode():
                validation_loss = float(
                    loss_fn(
                        model(X_validation_tensor),
                        y_validation_tensor,
                    ).item()
                )

            if (
                best_validation_loss is None
                or validation_loss
                < best_validation_loss - early_stopping_min_delta
            ):
                best_validation_loss = validation_loss
                best_epoch = epoch + 1
                best_state_dict = deepcopy(model.state_dict())
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1

        if logger is not None and (epoch == 0 or (epoch + 1) % 10 == 0):
            message = (
                f'Epoch {epoch + 1}/{num_epochs}, '
                f'train_loss={average_training_loss:.6f}'
            )
            if validation_loss is not None:
                message += f', validation_loss={validation_loss:.6f}'
            logger(message)

        if (
            validation_tensors is not None
            and early_stopping_patience is not None
            and early_stopping_patience > 0
            and epochs_without_improvement >= early_stopping_patience
        ):
            if best_state_dict is not None:
                model.load_state_dict(best_state_dict)

            if logger is not None:
                logger(
                    'Early stopping triggered at '
                    f'epoch {epoch + 1}/{num_epochs}. '
                    f'Best validation_loss={best_validation_loss:.6f} '
                    f'was reached at epoch {best_epoch}.'
                )
            return epoch + 1, best_validation_loss

    if validation_tensors is not None and best_state_dict is not None:
        model.load_state_dict(best_state_dict)
        if logger is not None and best_validation_loss is not None:
            logger(
                'Training completed without early stop. Restored the best '
                f'validation checkpoint from epoch {best_epoch} '
                f'with validation_loss={best_validation_loss:.6f}.'
            )

    return num_epochs, best_validation_loss


def _predict_scaled(
    model: '_ForecastingMLP',
    X: np.ndarray,
) -> np.ndarray:
    """Runs prediction without undoing target scaling.

    Args:
        model: Forecasting network.
        X: Feature matrix already scaled to the model input space.

    Returns:
        Predicted targets in scaled space.
    """

    X_tensor = torch.tensor(X, dtype=torch.float32)
    was_training = model.training
    model.eval()
    with torch.inference_mode():
        predictions = model(X_tensor).cpu().numpy()

    if was_training:
        model.train()

    return predictions


class _ForecastingMLP(nn.Module):
    """Implements a simple multilayer perceptron for regression."""

    def __init__(self, input_dim: int) -> None:
        """Initializes the network.

        Args:
            input_dim: Number of input features.
        """

        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.LayerNorm(32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.LayerNorm(16),
            nn.ReLU(),
            nn.Linear(16, 1),
        )
        self.x_scaler: StandardScaler | None = None
        self.y_scaler: StandardScaler | None = None
        self.target_transform: TargetTransformName = 'none'

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Runs a forward pass through the network.

        Args:
            x: Input tensor.

        Returns:
            Predicted target tensor.
        """

        return self.network(x)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predicts target values for the given feature matrix.

        Args:
            X: Feature matrix.

        Returns:
            Predicted target values on the original target scale.
        """

        if self.x_scaler is None or self.y_scaler is None:
            raise ValueError(
                'Scalers are not initialized. Train the model first.'
            )

        X_scaled = self.x_scaler.transform(X)
        X_tensor = torch.tensor(X_scaled, dtype=torch.float32)

        was_training = self.training
        self.eval()
        with torch.inference_mode():
            y_scaled = self(X_tensor).cpu().numpy()

        if was_training:
            self.train()

        y_model_space = self.y_scaler.inverse_transform(y_scaled).reshape(-1)
        return _inverse_transform_targets(
            y_model_space,
            self.target_transform,
        )


class NNForecastingExperiment(RegressionExperiment):
    def _get_stage_context_value(
        self,
        key: str,
        default: object,
        stage_name: str | None = None,
    ) -> object:
        """Returns a training setting, optionally overridden per stage.

        Args:
            key: Base context key to read.
            default: Fallback value when the key is absent.
            stage_name: Optional training stage prefix such as
                ``"pretraining"``.

        Returns:
            The resolved context value.
        """

        if stage_name is not None:
            stage_key = f'{stage_name}.{key}'
            if stage_key in self.context:
                return self.context[stage_key]

        return self.context.get(key, default)

    def _get_stage_hyperparameter(
        self,
        key: str,
        default: int | float,
        stage_name: str | None = None,
    ) -> int | float:
        """Returns a training hyperparameter, optionally overridden per stage.

        Args:
            key: Hyperparameter name to read.
            default: Fallback value when the hyperparameter is absent.
            stage_name: Optional training stage prefix such as
                ``"pretraining"``.

        Returns:
            The resolved hyperparameter value.
        """

        if stage_name is not None:
            stage_key = f'{stage_name}.{key}'
            if stage_key in self.context:
                value = self.context[stage_key]
                return cast(int | float, value)

        if key in self.hyperparameters:
            value = self.hyperparameters[key]
            return cast(int | float, value)

        value = self.context.get(key, default)
        return cast(int | float, value)

    def _prepare_model_for_training(
        self,
        input_dim: int,
        target_transform: TargetTransformName,
        existing_model: '_ForecastingMLP | None' = None,
    ) -> '_ForecastingMLP':
        """Builds or reuses a forecasting model for a training stage.

        Args:
            input_dim: Number of features in the current dataset.
            target_transform: Target transform to apply in this stage.
            existing_model: Optional model whose weights should be reused.

        Returns:
            The model to train for this stage.

        Raises:
            ValueError: If the existing model is incompatible with the
                provided input dimension.
        """

        if existing_model is not None:
            first_layer = existing_model.network[0]
            model_input_dim = getattr(first_layer, 'in_features', None)
            if model_input_dim != input_dim:
                raise ValueError(
                    'Cannot fine-tune NNForecastingExperiment with a '
                    'different number of input features.'
                )
            model = existing_model
        else:
            model = _ForecastingMLP(input_dim=input_dim)

        # Refit scalers for each stage while preserving the learned weights.
        model.x_scaler = StandardScaler()
        model.y_scaler = StandardScaler()
        model.target_transform = target_transform
        return model

    def _fit_training_stage(
        self,
        model: '_ForecastingMLP | None',
        X_train: np.ndarray,
        y_train: np.ndarray,
        stage_name: str,
    ) -> tuple['_ForecastingMLP', int, float | None]:
        """Fits one training stage and returns the trained model.

        Args:
            model: Optional model whose weights should be reused.
            X_train: Feature matrix for this stage.
            y_train: Target vector for this stage.
            stage_name: Human-readable stage name used for logging and
                stage-specific overrides.

        Returns:
            The trained model, number of epochs trained, and best validation
            loss when validation is available.
        """

        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        target_transform = _resolve_target_transform(
            self.context.get('target_transform')
        )

        validation_fraction = float(
            self._get_stage_context_value(
                'validation_fraction',
                0.2,
                stage_name=stage_name,
            )
        )
        early_stopping_patience = int(
            self._get_stage_context_value(
                'early_stopping.patience',
                25,
                stage_name=stage_name,
            )
        )
        early_stopping_min_delta = float(
            self._get_stage_context_value(
                'early_stopping.min_delta',
                1e-4,
                stage_name=stage_name,
            )
        )
        batch_size = int(
            self._get_stage_hyperparameter(
                'batch_size',
                64,
                stage_name=stage_name,
            )
        )
        learning_rate = float(
            self._get_stage_hyperparameter(
                'learning_rate',
                1e-3,
                stage_name=stage_name,
            )
        )
        weight_decay = float(
            self._get_stage_hyperparameter(
                'weight_decay',
                1e-4,
                stage_name=stage_name,
            )
        )
        num_epochs = int(
            self._get_stage_hyperparameter(
                'num_epochs',
                self.context.get('num_epochs', 500),
                stage_name=stage_name,
            )
        )

        model = self._prepare_model_for_training(
            input_dim=X_train.shape[1],
            target_transform=target_transform,
            existing_model=model,
        )
        x_scaler = cast(StandardScaler, model.x_scaler)
        y_scaler = cast(StandardScaler, model.y_scaler)

        self.log(
            f'{stage_name.capitalize()}: training neural network with '
            f'batch_size={batch_size}, '
            f'learning_rate={learning_rate}, '
            f'weight_decay={weight_decay}, '
            f'num_epochs={num_epochs}, '
            f'target_transform={target_transform}.'
        )

        X_fit = X_train
        y_fit = y_train.reshape(-1, 1)
        validation_data: tuple[np.ndarray, np.ndarray] | None = None
        if early_stopping_patience > 0 and X_train.shape[0] >= 3:
            (
                X_subtrain,
                y_subtrain,
                X_validation,
                y_validation,
            ) = _split_training_validation(
                X_train,
                y_train,
                validation_fraction=validation_fraction,
                seed=self.seed,
            )
            X_fit = X_subtrain
            y_fit = _transform_targets(
                y_subtrain,
                target_transform,
            ).reshape(-1, 1)

            X_fit_scaled = x_scaler.fit_transform(X_fit)
            y_fit_scaled = y_scaler.fit_transform(y_fit)
            y_validation_model = _transform_targets(
                y_validation,
                target_transform,
            )
            validation_data = (
                cast(np.ndarray, x_scaler.transform(X_validation)),
                cast(
                    np.ndarray,
                    y_scaler.transform(y_validation_model.reshape(-1, 1)),
                ),
            )
            self.log(
                f'{stage_name.capitalize()}: using early stopping with '
                f'{X_subtrain.shape[0]} subtraining samples, '
                f'{X_validation.shape[0]} validation samples, '
                f'patience={early_stopping_patience}, '
                f'min_delta={early_stopping_min_delta:.6f}.'
            )
        else:
            X_fit_scaled = x_scaler.fit_transform(X_fit)
            y_fit = _transform_targets(
                y_fit.reshape(-1),
                target_transform,
            ).reshape(-1, 1)
            y_fit_scaled = y_scaler.fit_transform(y_fit)
            if early_stopping_patience <= 0:
                self.log(
                    f'{stage_name.capitalize()}: early stopping disabled '
                    'because patience <= 0.'
                )
            elif X_train.shape[0] < 3:
                self.log(
                    f'{stage_name.capitalize()}: early stopping disabled '
                    'because the training split is too small for a '
                    'validation set.'
                )

        dataloader = _create_dataloader(
            X_fit_scaled,
            y_fit_scaled,
            batch_size=batch_size,
            seed=self.seed,
            shuffle=True,
        )

        trained_epochs, best_validation_loss = _fit_model(
            model,
            dataloader,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            num_epochs=num_epochs,
            validation_data=validation_data,
            early_stopping_patience=early_stopping_patience,
            early_stopping_min_delta=early_stopping_min_delta,
            logger=lambda message: self.log(
                f'{stage_name.capitalize()}: {message}'
            ),
        )
        return model, trained_epochs, best_validation_loss

    @override
    def tune(self) -> None:
        X_train, y_train = self.get_training_set()
        target_transform = _resolve_target_transform(
            self.context.get('target_transform')
        )

        default_num_epochs = int(self.context.get('num_epochs', 500))
        validation_fraction = float(
            self.context.get('validation_fraction', 0.2)
        )
        tuning_num_epochs = int(
            self.context.get('tuning.num_epochs', min(100, default_num_epochs))
        )

        if X_train.shape[0] < 3:
            self.hyperparameters = {
                'batch_size': 64,
                'learning_rate': 1e-3,
                'num_epochs': default_num_epochs,
                'weight_decay': 1e-4,
            }
            self.log(
                'Skipping neural network tuning because the training split is '
                'too small for a separate validation set.'
            )
            return

        (
            X_subtrain,
            y_subtrain,
            X_validation,
            y_validation,
        ) = _split_training_validation(
            X_train,
            y_train,
            validation_fraction=validation_fraction,
            seed=self.seed,
        )

        candidate_configs = [
            {
                'batch_size': 32,
                'learning_rate': 3e-4,
                'num_epochs': tuning_num_epochs,
                'weight_decay': 0.0,
            },
            {
                'batch_size': 32,
                'learning_rate': 1e-3,
                'num_epochs': tuning_num_epochs,
                'weight_decay': 1e-4,
            },
            {
                'batch_size': 64,
                'learning_rate': 3e-4,
                'num_epochs': tuning_num_epochs,
                'weight_decay': 1e-4,
            },
            {
                'batch_size': 64,
                'learning_rate': 1e-3,
                'num_epochs': tuning_num_epochs,
                'weight_decay': 1e-3,
            },
        ]

        self.log('Tuning neural network with candidate trials.')

        best_config: dict[str, int | float] | None = None
        best_validation_loss = float('inf')

        for config in candidate_configs:
            x_scaler = StandardScaler()
            y_scaler = StandardScaler()

            X_subtrain_scaled = x_scaler.fit_transform(X_subtrain)
            y_subtrain_model = _transform_targets(
                y_subtrain,
                target_transform,
            )
            y_subtrain_scaled = y_scaler.fit_transform(
                y_subtrain_model.reshape(-1, 1)
            )
            X_validation_scaled = x_scaler.transform(X_validation)
            y_validation_model = _transform_targets(
                y_validation,
                target_transform,
            )
            y_validation_scaled = y_scaler.transform(
                y_validation_model.reshape(-1, 1)
            )

            model = _ForecastingMLP(input_dim=X_train.shape[1])
            dataloader = _create_dataloader(
                X_subtrain_scaled,
                y_subtrain_scaled,
                batch_size=int(config['batch_size']),
                seed=self.seed,
                shuffle=True,
            )
            _fit_model(
                model,
                dataloader,
                learning_rate=float(config['learning_rate']),
                weight_decay=float(config['weight_decay']),
                num_epochs=int(config['num_epochs']),
            )

            validation_predictions = _predict_scaled(
                model,
                X_validation_scaled,  # type: ignore[assignment]
            )
            validation_loss = float(
                np.mean((validation_predictions - y_validation_scaled) ** 2)
            )
            self.log(
                'Tuning config '
                f'lr={config["learning_rate"]}, '
                f'weight_decay={config["weight_decay"]}, '
                f'batch_size={config["batch_size"]} '
                f'produced validation_loss={validation_loss:.6f}'
            )
            if validation_loss < best_validation_loss:
                best_validation_loss = validation_loss
                best_config = config

        if best_config is None:
            raise ValueError('No valid hyperparameter trial was found.')

        self.hyperparameters = {
            **best_config,
            'num_epochs': default_num_epochs,
        }
        self.log(
            'Selected neural network hyperparameters: '
            f'{self.hyperparameters} '
            f'(validation_loss={best_validation_loss:.6f})'
        )

    @override
    def train(self) -> None:
        X_train, y_train = self.get_training_set()
        super().train()

        model: _ForecastingMLP | None = None
        pretraining_dataset = cast(
            Dataset | None,
            self.context.get('pretraining_dataset'),
        )
        if pretraining_dataset is not None:
            y_pretraining = pretraining_dataset.y
            if y_pretraining is None:
                raise ValueError(
                    'Pretraining dataset must include target values.'
                )

            model, pretraining_epochs, pretraining_validation_loss = (
                self._fit_training_stage(
                    model=None,
                    X_train=pretraining_dataset.X,
                    y_train=y_pretraining,
                    stage_name='pretraining',
                )
            )
            self.parameters['pretraining_trained_num_epochs'] = (
                pretraining_epochs
            )
            if pretraining_validation_loss is not None:
                self.parameters['pretraining_best_validation_loss'] = (
                    pretraining_validation_loss
                )

        model, trained_epochs, best_validation_loss = self._fit_training_stage(
            model=model,
            X_train=X_train,
            y_train=y_train,
            stage_name=(
                'finetuning'
                if pretraining_dataset is not None
                else 'training'
            ),
        )
        self.parameters['trained_num_epochs'] = trained_epochs
        if best_validation_loss is not None:
            self.parameters['best_validation_loss'] = best_validation_loss

        self.model = model

    @override
    def evaluate(self) -> None:
        super().evaluate()


class NNForecastingTask(Task):
    @override
    def run(self, dataset: Dataset, schema: DatasetSchema) -> None:
        super().run(dataset)

        X, y = dataset.X, dataset.y
        assert y is not None

        seed = int(self.context.get('seed', 42))
        self.logger.info(f'Using seed {seed}.')

        for experiment in self.experiments:
            experiment = cast(NNForecastingExperiment, experiment)

            # Set the random seed for reproducibility.
            experiment.seed = seed

            # Inject schema into the experiment context.
            experiment.context['schema'] = schema

            experiment.setup(X, y, **self.context)
            experiment.run()
            experiment.logger.info(experiment.get_metrics())

    @override
    def check_dataset(self, dataset: Dataset) -> None:
        if dataset.y is None:
            raise ValueError('Dataset must have target values for regression.')
