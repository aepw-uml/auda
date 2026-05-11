from dataclasses import dataclass
from typing import Literal

RegressionMetricName = Literal['mae', 'rmse', 'r2', 'wape', 'mape']
REGRESSION_METRIC_NAMES: tuple[RegressionMetricName, ...] = (
    'mae',
    'rmse',
    'r2',
    'wape',
    'mape',
)

_REGRESSION_METRIC_LABELS: dict[RegressionMetricName, str] = {
    'mae': 'MAE',
    'rmse': 'RMSE',
    'r2': 'R²',
    'wape': 'WAPE',
    'mape': 'MAPE',
}


@dataclass(frozen=True)
class RegressionMetrics:
    """Container for regression evaluation metrics.

    Attributes:
        mae: Mean Absolute Error.
        rmse: Root Mean Squared Error.
        r2: R-squared score.
        wape: Weighted Absolute Percentage Error.
        mape: Mean Absolute Percentage Error.
    """

    mae: float = 0.0
    rmse: float = 0.0
    r2: float = 0.0
    wape: float = 0.0
    mape: float = 0.0

    def __repr__(self) -> str:
        """Returns a compact string representation of the metric values."""

        [mae_str, rmse_str, r2_str, wape_str, mape_str] = self.item_strs()

        return (
            f'Metrics(mae={mae_str}, rmse={rmse_str}, '
            f'r2={r2_str}, wape={wape_str}, mape={mape_str})'
        )

    def item_strs(self) -> list[str]:
        """Formats metric values as MAE, RMSE, R², WAPE, and MAPE."""

        return [
            f'{self.mae:.3e}'.replace('e+', 'e'),
            f'{self.rmse:.3e}'.replace('e+', 'e'),
            f'{self.r2:.3f}',
            f'{self.wape * 100:.2f}%',
            f'{self.mape * 100:.2f}%',
        ]

    def get_value_by_name(self, name: RegressionMetricName) -> float:
        """Returns the value of the specified metric.

        Args:
            name: The name of the metric to retrieve.

        Returns:
            The value of the specified metric.
        """

        match name:
            case 'mae':
                return self.mae
            case 'rmse':
                return self.rmse
            case 'r2':
                return self.r2
            case 'wape':
                return self.wape
            case 'mape':
                return self.mape


def average_regression_metrics(
    all_regression_metrics: list[RegressionMetrics],
) -> RegressionMetrics:
    """Averages a list of RegressionMetrics objects.

    Args:
        all_regression_metrics: A list of RegressionMetrics objects to average.

    Returns:
        A new RegressionMetrics object containing the average of each metric.
    """

    if not all_regression_metrics:
        raise ValueError('Cannot average an empty metrics list.')

    n = len(all_regression_metrics)
    return RegressionMetrics(
        mae=sum(m.mae for m in all_regression_metrics) / n,
        rmse=sum(m.rmse for m in all_regression_metrics) / n,
        r2=sum(m.r2 for m in all_regression_metrics) / n,
        wape=sum(m.wape for m in all_regression_metrics) / n,
        mape=sum(m.mape for m in all_regression_metrics) / n,
    )


def std_regression_metrics(
    all_regression_metrics: list[RegressionMetrics],
) -> RegressionMetrics:
    """Calculates sample standard deviations for regression metrics.

    Args:
        all_regression_metrics: A list of RegressionMetrics objects.

    Returns:
        A new RegressionMetrics object containing the sample standard deviation
        of each metric. Metrics with only one observation receive a standard
        deviation of zero.

    Raises:
        ValueError: If the provided metrics list is empty.
    """

    if not all_regression_metrics:
        raise ValueError('Cannot calculate standard deviation of empty list.')

    if len(all_regression_metrics) == 1:
        return RegressionMetrics()

    means = average_regression_metrics(all_regression_metrics)

    def sample_std(name: RegressionMetricName) -> float:
        values = [
            metrics.get_value_by_name(name)
            for metrics in all_regression_metrics
        ]
        mean = means.get_value_by_name(name)
        variance = sum((value - mean) ** 2 for value in values) / (
            len(values) - 1
        )
        return variance**0.5

    return RegressionMetrics(
        mae=sample_std('mae'),
        rmse=sample_std('rmse'),
        r2=sample_std('r2'),
        wape=sample_std('wape'),
        mape=sample_std('mape'),
    )


def get_regression_metric_label(name: RegressionMetricName) -> str:
    """Returns the display label for a regression metric.

    Args:
        name: Regression metric name.

    Returns:
        The display label for the metric.
    """

    return _REGRESSION_METRIC_LABELS[name]


def format_regression_metric_value(
    name: RegressionMetricName, value: float
) -> str:
    """Formats one regression metric value for tables or plots.

    Args:
        name: Regression metric name.
        value: Metric value.

    Returns:
        A display string for the metric value.
    """

    match name:
        case 'mae' | 'rmse':
            return f'{value:.3e}'.replace('e+', 'e')
        case 'r2':
            return f'{value:.3f}'
        case 'wape' | 'mape':
            return f'{value * 100:.2f}%'


def format_regression_metric_summary(
    name: RegressionMetricName,
    mean: float,
    std: float | None = None,
) -> str:
    """Formats a regression metric as a mean and optional standard deviation.

    Args:
        name: Regression metric name.
        mean: Mean metric value.
        std: Optional metric standard deviation.

    Returns:
        A display string containing the metric label, mean, and standard
        deviation when provided.
    """

    label = get_regression_metric_label(name)
    mean_str = format_regression_metric_value(name, mean)
    if std is None:
        return f'{label}: {mean_str}'

    std_str = format_regression_metric_value(name, std)
    return f'{label}: {mean_str} $\\pm$ {std_str}'
