import warnings
from itertools import product
from typing import Any, Self, override

import numpy as np
from statsmodels.tsa.arima.model import ARIMA
from step.model.model import SupervisedLearningModel


def _get_bool(value: Any, default: bool = False) -> bool:
    """Converts common boolean-like values to a boolean.

    Args:
        value: Value to convert.
        default: Value to return when ``value`` is None.

    Returns:
        The converted boolean value.
    """

    if value is None:
        return default

    if isinstance(value, bool):
        return value

    if isinstance(value, str):
        return value.lower() not in {'0', 'false', 'no', 'off'}

    return bool(value)


class ARIMARegression(SupervisedLearningModel):
    """Forecasts one-dimensional time series with an ARIMA model."""

    def __init__(
        self,
        hyperparameters: dict[str, Any],
        **kwargs,
    ) -> None:
        """Initializes the ARIMA regression model.

        Args:
            hyperparameters: Model configuration containing fixed ``p``,
                ``d``, and ``q`` values, or auto-ARIMA search limits.
            **kwargs: Additional keyword arguments forwarded to the base
                class.
        """

        super().__init__(hyperparameters, **kwargs)
        default_auto = not {'p', 'd', 'q'}.issubset(hyperparameters)
        self.hyperparameters: dict[str, Any] = {
            'auto': _get_bool(
                hyperparameters.get(
                    'auto',
                    hyperparameters.get('auto_arima', default_auto),
                )
            ),
            'p': int(hyperparameters.get('p', 2)),
            'd': int(hyperparameters.get('d', 0)),
            'q': int(hyperparameters.get('q', 1)),
            'max_p': int(hyperparameters.get('max_p', 2)),
            'max_d': int(hyperparameters.get('max_d', 1)),
            'max_q': int(hyperparameters.get('max_q', 2)),
            'max_order': int(hyperparameters.get('max_order', 5)),
            'information_criterion': str(
                hyperparameters.get('information_criterion', 'aic')
            ),
            'trend': str(hyperparameters.get('trend', 'n')),
        }

    @override
    def fit(self, X: np.ndarray, y: np.ndarray) -> Self:
        """Fits the ARIMA model on the sorted time series.

        Args:
            X: Training time indices with shape ``(n_samples, 1)``.
            y: Training targets with shape ``(n_samples,)``.

        Returns:
            The fitted estimator.

        Raises:
            ValueError: If the ARIMA orders are invalid.
        """

        super().fit(X, y, num_features=1)

        order = np.argsort(X[:, 0])
        X_sorted = X[order, 0]
        y_sorted = y[order]

        if self.hyperparameters['auto']:
            arima_order, fitted_model = self._fit_auto_arima(y_sorted)
            self.parameters['selected_order'] = arima_order
            self.hyperparameters['selected_order'] = arima_order
        else:
            p = self.hyperparameters['p']
            d = self.hyperparameters['d']
            q = self.hyperparameters['q']
            arima_order = (p, d, q)
            self._validate_order(arima_order)
            fitted_model = self._fit_arima(y_sorted, arima_order)

        self.parameters['last_x'] = float(X_sorted[-1])
        self.parameters['order'] = arima_order
        self.parameters['trend'] = self.hyperparameters['trend']
        self.model_ = fitted_model
        self.parameters['aic'] = float(self.model_.aic)

        return self

    def _fit_auto_arima(self, y: np.ndarray) -> tuple[tuple[int, int, int], Any]:
        """Selects and fits the best ARIMA order by information criterion.

        Args:
            y: Sorted training target values.

        Returns:
            The selected ARIMA order and its fitted statsmodels result.

        Raises:
            ValueError: If search limits are invalid or no candidate fits.
        """

        max_p = self.hyperparameters['max_p']
        max_d = self.hyperparameters['max_d']
        max_q = self.hyperparameters['max_q']
        max_order = self.hyperparameters['max_order']
        information_criterion = self.hyperparameters['information_criterion']
        if min(max_p, max_d, max_q, max_order) < 0:
            raise ValueError('Auto-ARIMA search limits must be non-negative.')

        if information_criterion not in {'aic', 'bic', 'hqic'}:
            raise ValueError(
                'Auto-ARIMA information criterion must be "aic", "bic", '
                'or "hqic".'
            )

        best_order: tuple[int, int, int] | None = None
        best_model: Any | None = None
        best_score = np.inf
        for p, d, q in product(
            range(max_p + 1),
            range(max_d + 1),
            range(max_q + 1),
        ):
            arima_order = (p, d, q)
            if sum(arima_order) > max_order:
                continue

            try:
                fitted_model = self._fit_arima(y, arima_order)
                score = float(getattr(fitted_model, information_criterion))
            except Exception:
                continue

            if np.isfinite(score) and score < best_score:
                best_order = arima_order
                best_model = fitted_model
                best_score = score

        if best_order is None or best_model is None:
            raise ValueError('Auto-ARIMA could not fit any candidate order.')

        return best_order, best_model

    def _fit_arima(self, y: np.ndarray, order: tuple[int, int, int]) -> Any:
        """Fits one ARIMA model while suppressing candidate-search warnings.

        Args:
            y: Sorted training target values.
            order: ARIMA ``(p, d, q)`` order to fit.

        Returns:
            The fitted statsmodels ARIMA result.
        """

        self._validate_order(order)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            return ARIMA(
                y,
                order=order,
                trend=self.hyperparameters['trend'],
            ).fit()

    def _validate_order(self, order: tuple[int, int, int]) -> None:
        """Validates that an ARIMA order contains non-negative integers.

        Args:
            order: ARIMA ``(p, d, q)`` order to validate.

        Raises:
            ValueError: If the order contains a negative value.
        """

        if min(order) < 0:
            raise ValueError('ARIMA orders p, d, and q must be non-negative.')

    @override
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Forecasts future values at the requested time steps.

        Args:
            X: Future time indices with shape ``(n_samples, 1)``.

        Returns:
            Predicted targets with shape ``(n_samples,)``.

        Raises:
            ValueError: If requested periods are not whole future time steps
                or are not in the future.
        """

        super().predict(X, num_features=1)

        X = np.asarray(X, dtype=float)
        t: np.ndarray = X[:, 0]

        steps = t - self.parameters['last_x']
        if not np.allclose(steps, np.round(steps)):
            raise ValueError(
                'ARIMARegression expects forecast periods aligned to whole '
                'time steps.'
            )

        steps = np.round(steps).astype(int)
        if np.any(steps <= 0):
            raise ValueError(
                'ARIMARegression can only forecast future time steps.'
            )

        max_step: int = int(steps.max())
        forecast = self.model_.forecast(steps=max_step)
        y_hat = forecast[steps - 1]

        return np.asarray(y_hat, dtype=float)
