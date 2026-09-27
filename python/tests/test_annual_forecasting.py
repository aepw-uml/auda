"""Checks calendar gaps and training-only handling in annual forecasters."""

import unittest
from unittest.mock import patch

import numpy as np
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.holtwinters import ExponentialSmoothing as Holt
from step.model.annual_series import annual_training_series
from step.model.arima_regression import ARIMARegression
from step.model.exponential_smoothing import ExponentialSmoothing


class AnnualForecastingTest(unittest.TestCase):
    """Checks missing-year representation and unchanged complete-series fits."""

    def test_grid_preserves_years_and_inputs(self) -> None:
        """Sorts observations and inserts gaps without modifying input arrays."""

        X = np.array([[1976], [1973], [1975]])
        y = np.array([54., 51., 46.])
        years, values = annual_training_series(X, y)
        np.testing.assert_array_equal(years, [1973, 1974, 1975, 1976])
        np.testing.assert_allclose(values, [51, np.nan, 46, 54])
        np.testing.assert_array_equal(y, [54, 51, 46])
        for times in ([[1973], [1973]], [[1973], [1974.5]]):
            with self.assertRaises(ValueError):
                annual_training_series(np.array(times), np.array([1., 2.]))

    def test_arima_candidates_preserve_missing_position(self) -> None:
        """Checks every candidate receives NaN at the absent training year."""

        years = np.arange(1960, 1980)
        values = 100 + 2 * np.arange(20) + np.sin(np.arange(20))
        keep = years != 1974
        model = ARIMARegression({'max_p': 1, 'max_d': 1, 'max_q': 1})
        original_fit = model._fit_arima
        candidates = []

        def inspect_candidate(y, order):
            """Records candidate inputs while running actual model fitting."""

            candidates.append(order)
            self.assertEqual(len(y), 20)
            self.assertTrue(np.isnan(y[14]))
            return original_fit(y, order)

        with patch.object(model, '_fit_arima', side_effect=inspect_candidate):
            model.fit(years[keep, None], values[keep])
        self.assertEqual(len(candidates), 8)
        self.assertEqual(model.parameters['missing_years'], [1974])
        self.assertEqual(model.model_.nobs, 20)
        np.testing.assert_allclose(
            model.predict(np.array([[1980], [1982]])),
            model.model_.forecast(3)[[0, 2]],
        )

    def test_holt_interpolates_only_inside_training_years(self) -> None:
        """Checks interpolation ends at the training cutoff before forecasting."""

        years = np.arange(2000, 2012)
        values = 100 + 2 * np.arange(12) + np.sin(np.arange(12))
        keep = years != 2004
        model = ExponentialSmoothing({}).fit(years[keep, None], values[keep])
        expected = values.copy()
        expected[4] = (values[3] + values[5]) / 2
        np.testing.assert_allclose(model.model_.model.endog, expected)
        self.assertEqual(model.parameters['last_x'], 2011)
        self.assertEqual(model.parameters['missing_years'], [2004])
        np.testing.assert_allclose(
            model.predict(np.array([[2012], [2014]])),
            model.model_.forecast(3)[[0, 2]],
        )

    def test_complete_series_matches_original_estimators(self) -> None:
        """Checks that the preprocessing preserves complete-series forecasts."""

        years = np.arange(2000, 2020).reshape(-1, 1)
        values = 100 + 2 * np.arange(20) + np.sin(np.arange(20))
        future = np.array([[2020], [2021]])
        arima = ARIMARegression({'p': 1, 'd': 1, 'q': 0}).fit(years, values)
        reference = ARIMA(values, order=(1, 1, 0), trend='n').fit()
        np.testing.assert_allclose(arima.predict(future), reference.forecast(2))
        holt = ExponentialSmoothing({}).fit(years, values)
        reference = Holt(
            values, trend='add', seasonal=None, initialization_method='estimated'
        ).fit()
        np.testing.assert_allclose(holt.predict(future), reference.forecast(2))


if __name__ == '__main__':
    unittest.main()
