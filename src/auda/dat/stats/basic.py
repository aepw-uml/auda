from typing import List, override

from auda.dat.datasets import LabeledSamples
from auda.utils.pipeline import IOSpec, Task, task

from .__common import STAT_KIND, StatISName, StatOSName


@task(
    id='ST-BASIC',
    kind=STAT_KIND,
    description='Computes basic descriptive statistics of the input samples.',
    input_specs={
        StatISName.SAMPLES: IOSpec(dtype=LabeledSamples),
    },
    output_specs={
        StatOSName.NUM_SAMPLES: IOSpec(dtype=int),
        StatOSName.NUM_FEATURES: IOSpec(dtype=int),
        StatOSName.X_MEAN: IOSpec(dtype=List[float]),
        StatOSName.X_MINIMUM: IOSpec(dtype=List[float]),
        StatOSName.X_MAXIMUM: IOSpec(dtype=List[float]),
        StatOSName.X_STANDARD_DEVIATION: IOSpec(dtype=List[float]),
        StatOSName.X_VARIANCE: IOSpec(dtype=List[float]),
        StatOSName.Y_MEAN: IOSpec(dtype=float),
        StatOSName.Y_MINIMUM: IOSpec(dtype=float),
        StatOSName.Y_MAXIMUM: IOSpec(dtype=float),
        StatOSName.Y_STANDARD_DEVIATION: IOSpec(dtype=float),
        StatOSName.Y_VARIANCE: IOSpec(dtype=float),
    },
)
class BasicStats(Task):
    @override
    def run(self) -> None:
        """
        Compute basic statistics of the dataset.
        """
        import numpy as np

        # Data
        samples = self.get_input(StatISName.SAMPLES)
        x = np.array([x for x, _ in samples])
        y = np.array([y for _, y in samples])

        # Statistics
        num_samples: int = len(x)
        num_features: int = len(x[0])
        x_mean: List[float] = np.mean(x, axis=0).tolist()
        x_minimum: List[float] = np.min(x, axis=0).tolist()
        x_maximum: List[float] = np.max(x, axis=0).tolist()
        x_standard_deviation: List[float] = np.std(x, axis=0).tolist()
        x_variance: List[float] = np.var(x, axis=0).tolist()

        # y is 1D → scalars
        y_mean: float = float(y.mean())
        y_minimum: float = float(y.min())
        y_maximum: float = float(y.max())
        y_standard_deviation: float = float(y.std())  # ddof=0
        y_variance: float = float(y.var())

        # Output
        self.set_output(StatOSName.NUM_SAMPLES, num_samples)
        self.set_output(StatOSName.NUM_FEATURES, num_features)
        self.set_output(StatOSName.X_MEAN, x_mean)
        self.set_output(StatOSName.X_MINIMUM, x_minimum)
        self.set_output(StatOSName.X_MAXIMUM, x_maximum)
        self.set_output(StatOSName.X_STANDARD_DEVIATION, x_standard_deviation)
        self.set_output(StatOSName.X_VARIANCE, x_variance)
        self.set_output(StatOSName.Y_MEAN, y_mean)
        self.set_output(StatOSName.Y_MINIMUM, y_minimum)
        self.set_output(StatOSName.Y_MAXIMUM, y_maximum)
        self.set_output(StatOSName.Y_STANDARD_DEVIATION, y_standard_deviation)
        self.set_output(StatOSName.Y_VARIANCE, y_variance)

        # TODO: Extend StatsModel with richer descriptive and diagnostic statistics
        # -------------------------------------------------------------------------
        # 1. Feature-level metrics:
        #    - Median, quantiles (25%, 50%, 75%)
        #    - Skewness and kurtosis
        #    - Missing value counts and proportions
        #    - Outlier counts (e.g. |z| > 3)
        #    - Normality test (Shapiro-Wilk or D’Agostino)
        #
        # 2. Relationships between features:
        #    - Covariance matrix
        #    - Correlation matrix (Pearson / Spearman)
        #    - Pairwise feature correlations summary
        #
        # 3. Target variable diagnostics:
        #    - Distribution histogram (continuous y)
        #    - Class frequencies / imbalance ratio (categorical y)
        #    - Feature–target correlation or ANOVA scores
        #
        # 4. Dataset-level info:
        #    - Missingness summary (total and per feature)
        #    - Unique value counts (for categorical columns)
        #    - Data type summary (numeric vs categorical)
        #
        # 5. Optional:
        #    - PCA explained variance ratio (for high-dimensional data)
        #    - Feature scaling check (min-max, z-score)
        #    - Time-based trend analysis (if data is temporal)
