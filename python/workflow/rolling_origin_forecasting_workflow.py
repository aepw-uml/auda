"""Evaluates univariate forecasts at successive historical cutoffs."""

import csv
import json
from dataclasses import asdict
from pathlib import Path
from typing import cast, override

import numpy as np
from common.dataset import Dataset, DatasetSchema
from common.experiment.forecasting_experiment import ForecastingExperiment
from common.experiment.persistence import (
    save_gpr_fit_details,
    save_hyperparameter_table,
    save_metric_table,
)
from common.metrics import RegressionMetrics, average_regression_metrics
from common.metrics.regression_metrics import std_regression_metrics
from common.workflow import Workflow
from experiment.forecasting_task import run_forecasting_tasks
from util.names import to_snake


def rolling_origins(
    dataset: Dataset, initial_train_size: int, horizon: int, step: int
) -> list[int]:
    """Returns expanding training sizes with complete future test windows.

    Args:
        dataset: Univariate observations in strictly increasing time order.
        initial_train_size: Number of observations before the first forecast.
        horizon: Number of subsequent observations evaluated at each origin.
        step: Number of observations between origins.

    Returns:
        Training sizes for all complete forecast windows.

    Raises:
        ValueError: If data, settings, or available window counts are invalid.
    """

    X, y = dataset.X, dataset.y
    if (
        X.ndim != 2
        or X.shape[1] != 1
        or y is None
        or y.ndim != 1
        or len(X) != len(y)
        or not np.isfinite(X).all()
        or not np.isfinite(y).all()
    ):
        raise ValueError('Rolling origins require finite univariate data.')
    if not np.all(np.diff(X[:, 0]) > 0):
        raise ValueError('Observation times must be strictly increasing.')
    if initial_train_size < 8 or horizon < 1 or step < 1:
        raise ValueError(
            'initial_train_size must be >= 8; horizon and step must be >= 1. '
            'Eight observations provide at least two in the initial CV '
            'training block; validation blocks may contain one observation.'
        )
    origins = list(range(initial_train_size, len(y) - horizon + 1, step))
    if len(origins) < 2:
        raise ValueError('Settings must provide at least two complete origins.')
    return origins


def save_rows(path: Path, rows: list[dict]) -> None:
    """Writes records to a CSV file using the first record's field order.

    Args:
        path: Destination file.
        rows: Nonempty list of records with matching fields.
    """

    with path.open('w', newline='', encoding='utf-8') as output:
        writer = csv.DictWriter(output, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


class RollingOriginForecastingWorkflow(Workflow):
    """Refits and retunes forecasting models independently at each origin."""

    @override
    def run(self, dataset: Dataset, schema: DatasetSchema, **context) -> None:
        """Saves per-run predictions and equally weighted origin summaries.

        Args:
            dataset: Ordered univariate dataset.
            schema: Dataset names and units.
            **context: Required initial_train_size and horizon; optional step
                (default horizon), num_experiments (default 1), seed (42),
                workflow_name, and existing forecasting task options.
        """

        if 'initial_train_size' not in context or 'horizon' not in context:
            raise ValueError(
                'Specify initial_train_size and horizon explicitly.'
            )
        initial = int(context['initial_train_size'])
        horizon = int(context['horizon'])
        step = int(context.get('step', horizon))
        repetitions = int(context.get('num_experiments', 1))
        seed = int(context.get('seed', 42))
        if repetitions < 1:
            raise ValueError('num_experiments must be >= 1.')
        origins = rolling_origins(dataset, initial, horizon, step)
        assert dataset.y is not None
        location = to_snake(context.get('location', ''))
        name = context.get('workflow_name') or (
            'rolling_origin_forecasting' + (f'_{location}' if location else '')
        )
        destination = Path('results') / name
        destination.mkdir(parents=True, exist_ok=True)
        origin_metrics: dict[str, list[RegressionMetrics]] = {}
        metric_rows, prediction_rows, origin_rows = [], [], []
        for origin_index, train_size in enumerate(origins, start=1):
            run_context = dict(context)
            run_context['forecast_train_size'] = str(train_size)
            prefix = Dataset(
                dataset.X[: train_size + horizon],
                dataset.y[: train_size + horizon],
            )
            tasks, means = run_forecasting_tasks(
                repetitions, prefix, schema, run_context, seed=seed
            )
            origin_dir = destination / f'origin_{origin_index:03d}'
            save_metric_table(means, origin_dir)
            save_hyperparameter_table(tasks[0], origin_dir)
            save_gpr_fit_details(tasks, origin_dir)
            origin_time = float(dataset.X[train_size - 1, 0])
            for model_name, metrics in means.items():
                origin_metrics.setdefault(model_name, []).append(metrics)
                origin_rows.append(
                    {
                        'origin': origin_index,
                        'origin_time': origin_time,
                        'train_size': train_size,
                        'model': model_name,
                        **asdict(metrics),
                    }
                )
            for run_index, task in enumerate(tasks, start=1):
                for experiment in task.experiments:
                    metadata = {
                        'origin': origin_index,
                        'origin_time': origin_time,
                        'train_size': train_size,
                        'run': run_index,
                        'seed': experiment.seed,
                        'model': experiment.name,
                    }
                    metric_rows.append(
                        {
                            **metadata,
                            **asdict(experiment.get_metrics()),
                            'hyperparameters': json.dumps(
                                experiment.hyperparameters
                            ),
                        }
                    )
                    forecast_experiment = cast(
                        ForecastingExperiment, experiment
                    )
                    X_test, _ = forecast_experiment.get_test_set()
                    predictions = forecast_experiment.get_model().predict(
                        X_test
                    )
                    for offset, prediction in enumerate(predictions):
                        test_index = train_size + offset
                        test_time = float(dataset.X[test_index, 0])
                        prediction_rows.append(
                            {
                                **metadata,
                                'horizon_step': offset + 1,
                                'test_time': test_time,
                                'elapsed_time': test_time - origin_time,
                                'observed': float(dataset.y[test_index]),
                                'predicted': float(prediction),
                            }
                        )
        means = {
            name: average_regression_metrics(values)
            for name, values in origin_metrics.items()
        }
        save_metric_table(means, destination)
        save_metric_table(
            {
                name: std_regression_metrics(values)
                for name, values in origin_metrics.items()
            },
            destination / 'origin_std',
        )
        save_rows(destination / 'run_metrics.csv', metric_rows)
        save_rows(destination / 'origin_metrics.csv', origin_rows)
        save_rows(destination / 'predictions.csv', prediction_rows)
        manifest = {
            'initial_train_size': initial,
            'horizon': horizon,
            'step': step,
            'num_origins': len(origins),
            'num_experiments': repetitions,
            'seed': seed,
            'context': context,
            'schema': asdict(schema),
            'aggregation': (
                'Mean across seeds within origin, then across origins.'
            ),
            'dispersion': (
                'Sample SD of origin means; not a confidence interval.'
            ),
            'horizon_units': 'observations; elapsed_time records calendar gaps',
            'percentage_metrics': 'WAPE and sMAPE are fractions in CSV files.',
        }
        (destination / 'evaluation.json').write_text(
            json.dumps(manifest, indent=2), encoding='utf-8'
        )
