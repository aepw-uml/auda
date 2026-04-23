from dataclasses import dataclass
from typing import Literal

RegressionMetricName = Literal['mae', 'rmse', 'r2', 'wape', 'mape']


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

    n = len(all_regression_metrics)
    return RegressionMetrics(
        mae=sum(m.mae for m in all_regression_metrics) / n,
        rmse=sum(m.rmse for m in all_regression_metrics) / n,
        r2=sum(m.r2 for m in all_regression_metrics) / n,
        wape=sum(m.wape for m in all_regression_metrics) / n,
        mape=sum(m.mape for m in all_regression_metrics) / n,
    )
