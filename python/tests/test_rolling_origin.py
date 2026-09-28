"""Checks chronological isolation and rolling-origin result aggregation."""

import csv
import json
import tempfile
import unittest
from pathlib import Path
from typing import override
from unittest.mock import patch

import numpy as np
from common.dataset import Dataset, DatasetSchema
from experiment.forecasting_task import (
    ForecastingTask,
    get_naive_persistence_forecasting,
    get_ridge_regression_forecasting,
)
from workflow.rolling_origin_forecasting_workflow import (
    RollingOriginForecastingWorkflow,
    rolling_origins,
)


class RollingOriginTest(unittest.TestCase):
    """Checks test windows, tuning isolation, and saved forecasts."""

    @override
    def setUp(self) -> None:
        """Creates a small annual series with known persistence errors."""

        self.dataset = Dataset(
            np.arange(2000, 2012, dtype=float).reshape(-1, 1),
            np.arange(1, 13, dtype=float),
        )
        self.schema = DatasetSchema(['Year'], [''], ['Value'], ['tonnes'])

    def test_windows_and_invalid_inputs(self) -> None:
        """Rejects invalid chronology and excludes incomplete final windows."""

        self.assertEqual(rolling_origins(self.dataset, 8, 2, 1), [8, 9, 10])
        self.assertEqual(rolling_origins(self.dataset, 8, 3, 1), [8, 9])
        invalid_settings = [(7, 2, 1), (8, 0, 1), (8, 2, 0), (10, 2, 1)]
        for initial, horizon, step in invalid_settings:
            with self.assertRaises(ValueError):
                rolling_origins(self.dataset, initial, horizon, step)
        with self.assertRaises(ValueError):
            rolling_origins(
                Dataset(self.dataset.X[::-1], self.dataset.y), 8, 2, 1
            )

    def test_tuning_never_receives_outer_test_data(self) -> None:
        """Checks all inner tuning folds stay before the forecast cutoff."""

        experiment = get_ridge_regression_forecasting()
        assert self.dataset.y is not None
        experiment.setup(
            self.dataset.X[:10], self.dataset.y[:10], forecast_train_size='8'
        )
        experiment.split()
        from common.experiment.forecasting_experiment import (
            time_series_cross_validation,
        )

        seen = []

        def inspect_validation(X, y, evaluate):
            """Records real tuning partitions before evaluating each fold."""

            self.assertEqual(X[:, 0].tolist(), list(range(2000, 2008)))

            def inspect_fold(X_train, y_train, X_val, y_val):
                """Checks chronological order inside each real tuning fold."""

                self.assertLess(X_train[-1, 0], X_val[0, 0])
                self.assertLess(X_val[-1, 0], 2008)
                seen.append(len(X_train))
                return evaluate(X_train, y_train, X_val, y_val)

            return time_series_cross_validation(X, y, inspect_fold)

        experiment.context['tuning_parameters']['num_iterations'] = 1
        with patch(
            'common.experiment.forecasting_experiment.'
            'time_series_cross_validation',
            side_effect=inspect_validation,
        ):
            experiment.tune()
        experiment.train()
        self.assertEqual(seen, [2, 4, 6, 7])
        assert experiment.X_test is not None
        self.assertEqual(experiment.X_test[:, 0].tolist(), [2008, 2009])
        self.assertEqual(experiment.model.x_scaler_.mean_[0], 2003.5)

    def test_workflow_outputs_and_seed_aggregation(self) -> None:
        """Runs real persistence fits and verifies origin-level summaries."""

        def baseline_task(context):
            """Builds a small task while retaining the real execution path."""

            task = ForecastingTask(name='Test')
            task.set_context(**context)
            task.add(get_naive_persistence_forecasting())
            return task

        with tempfile.TemporaryDirectory() as temporary:
            with patch(
                'experiment.forecasting_task.get_forecasting_task',
                side_effect=baseline_task,
            ):
                RollingOriginForecastingWorkflow().run(
                    self.dataset,
                    self.schema,
                    initial_train_size='8',
                    horizon='2',
                    step='2',
                    num_experiments='2',
                    workflow_name=temporary,
                )
            path = Path(temporary)
            with (path / 'predictions.csv').open() as source:
                predictions = list(csv.DictReader(source))
            self.assertEqual(len(predictions), 8)
            for row in predictions:
                self.assertLess(
                    float(row['origin_time']), float(row['test_time'])
                )
                self.assertEqual(
                    float(row['predicted']), float(row['train_size'])
                )
            with (path / 'origin_metrics.csv').open() as source:
                metrics = list(csv.DictReader(source))
            self.assertEqual(len(metrics), 2)
            self.assertAlmostEqual(float(metrics[0]['wape']), 3 / 19)
            self.assertAlmostEqual(float(metrics[1]['wape']), 3 / 23)
            manifest = json.loads((path / 'evaluation.json').read_text())
            self.assertEqual(manifest['num_origins'], 2)
            self.assertTrue((path / 'origin_std' / 'metric_table').exists())


if __name__ == '__main__':
    unittest.main()
