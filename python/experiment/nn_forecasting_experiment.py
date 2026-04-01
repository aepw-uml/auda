from typing import override

import numpy as np
import torch
from common.experiment.regression_experiment import RegressionExperiment
from sklearn.preprocessing import StandardScaler
from torch import nn


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
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.Linear(16, 1),
        )
        self.x_scaler: StandardScaler | None = None
        self.y_scaler: StandardScaler | None = None

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

        self.eval()
        with torch.no_grad():
            y_scaled = self(X_tensor).cpu().numpy()

        return self.y_scaler.inverse_transform(y_scaled).reshape(-1)


class NNForecastingExperiment(RegressionExperiment):
    @override
    def tune(self) -> None:
        pass

    @override
    def train(self) -> None:
        X_train, y_train = self.get_training_set()
        super().train()

        torch.manual_seed(self.seed)
        np.random.seed(self.seed)

        x_scaler = StandardScaler()
        y_scaler = StandardScaler()

        X_train_scaled = x_scaler.fit_transform(X_train)
        y_train_scaled = y_scaler.fit_transform(y_train.reshape(-1, 1))

        model = _ForecastingMLP(input_dim=X_train.shape[1])
        model.x_scaler = x_scaler
        model.y_scaler = y_scaler

        dataset = torch.utils.data.TensorDataset(
            torch.tensor(X_train_scaled, dtype=torch.float32),
            torch.tensor(y_train_scaled, dtype=torch.float32),
        )
        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=64,
            shuffle=True,
            generator=torch.Generator().manual_seed(self.seed),
        )

        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=1e-3,
            weight_decay=1e-4,
        )
        loss_fn = nn.MSELoss()

        num_epochs = 500
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

            if (epoch + 1) % 100 == 0:
                self.log(
                    f'Epoch {epoch + 1}/{num_epochs}, '
                    f'loss={epoch_loss / num_batches:.6f}'
                )

        self.model = model

    @override
    def evaluate(self) -> None:
        super().evaluate()
