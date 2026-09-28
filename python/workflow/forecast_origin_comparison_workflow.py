"""Reports per-origin WAPE from completed rolling-origin forecasts."""

from pathlib import Path
from typing import override

import pandas as pd
from common.dataset import Dataset, DatasetSchema
from common.files import save_content_to_file
from common.workflow import Workflow
from util.table import Table


class ForecastOriginComparisonWorkflow(Workflow):
    """Compares four forecasting models without refitting saved experiments."""

    @override
    def run(self, dataset: Dataset, schema: DatasetSchema, **context) -> None:
        """Saves origin means and their equally weighted overall average.

        Args:
            dataset: Dataset supplied by the workflow command.
            schema: Dataset metadata supplied by the workflow command.
            **context: Required source_workflow naming completed results.
        """

        source = Path('results') / context['source_workflow']
        metrics = pd.read_csv(source / 'origin_metrics.csv')
        predictions = pd.read_csv(source / 'predictions.csv')
        models = [
            'Exponential Smoothing',
            'Theil-Sen Regression',
            'Ridge Regression',
            'Support Vector Regression',
        ]
        values = metrics.pivot(index='origin', columns='model', values='wape')
        values = values.loc[:, models].sort_index()
        if values.empty or values.isna().any().any():
            raise ValueError('Each origin must contain WAPE for all models.')
        table = Table(
            headers=[
                'Test years',
                'Exponential Smoothing',
                'Theil–Sen',
                'Ridge',
                'SVR',
            ]
        )
        for origin, row in values.iterrows():
            years = predictions.loc[
                predictions['origin'] == origin, 'test_time'
            ]
            if years.empty:
                raise ValueError(f'Missing test years for origin {origin}.')
            table.append_row(
                f'{years.min():g}–{years.max():g}',
                *[f'{100 * value:.2f}%' for value in row],
            )
        table.append_row(
            'Average across origins',
            *[f'{100 * value:.2f}%' for value in values.mean()],
        )
        destination = source / 'origin_comparison_table'
        save_content_to_file(destination, repr(table))
        print(table)
        print(f'Saved origin comparison to "{destination}".')
