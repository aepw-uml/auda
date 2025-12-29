from random import randrange
from typing import override

import numpy as np
from auda.step.dataset import DatasetBasedStep
from auda.step.spec import Dataset, Spec
from auda.utils.pipeline import IOValueMap, step


# TODO: Doesn't work currently; refactor it after asking professor's opinion
@step(
    id='SY-TS',
    description='Generate synthetic time-series data (optionally extending timeline).',
    input_specs=[
        Spec.ON.optional(Spec.DATASET.name),
        Spec.SEED.optional(),
        Spec.SYNTHETIC_SCALE.optional(1.0),
        Spec.TREND_STRENGTH.optional(0.0),
        Spec.SEASONAL_STRENGTH.optional(0.0),
        Spec.SEASONAL_PERIOD.optional(12),
        Spec.EXTRA_POINTS.optional(0),
        Spec.SORT_BY_X.optional(True),
    ],
    output_specs=[
        Spec.SYNTHETIC_DATASET,
        Spec.SEED,
    ],
)
class SyntheticTimeSeriesGenerator(DatasetBasedStep):
    @override
    def run(
        self,
        on: str | Dataset,
        seed: int | None,
        synthetic_scale: float,
        trend_strength: float,
        seasonal_strength: float,
        seasonal_period: int,
        extra_points: int,
        sort_by_x: bool,
    ) -> IOValueMap:
        if seed is None:
            seed = randrange(2**32)

        rng = np.random.default_rng(seed)

        X, y = self.get_dataset_from_on(on)
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)

        if X.ndim != 2 or X.shape[1] != 1:
            raise ValueError(
                f'Expected X to have shape (n, 1) for Year, got {X.shape}.'
            )

        # Sort by year if requested (your DS-YEAR-PW output is currently shuffled)
        if sort_by_x:
            order = np.argsort(X[:, 0])
            X = X[order]
            y = y[order]

        # Extend X into the future (or past, if extra_points is negative)
        if extra_points != 0:
            last_year = float(X[-1, 0])
            new_years = last_year + 1 * np.arange(
                1, extra_points + 1, dtype=float
            )
            X_extra = new_years.reshape(-1, 1)
            X_ext = np.vstack([X, X_extra])
        else:
            X_ext = X

        N = X_ext.shape[0]

        # Stats from real observations (only original y)
        mean = float(np.mean(y))
        std = float(np.std(y)) if float(np.std(y)) > 0 else 1.0

        # Synthetic series for the whole extended timeline
        t = np.arange(N, dtype=float)
        trend = trend_strength * t
        seasonal = seasonal_strength * np.sin(
            2 * np.pi * t / float(seasonal_period)
        )
        noise = rng.normal(0.0, std * float(synthetic_scale), size=N)

        y_synth = mean + trend + seasonal + noise

        # Keep non-negativity if your label is tonnage (optional but often sensible)
        y_synth = np.maximum(y_synth, 0.0)

        synthetic_dataset: Dataset = (X_ext, y_synth)

        return {
            Spec.SEED.name: seed,
            Spec.SYNTHETIC_DATASET.name: synthetic_dataset,
        }
