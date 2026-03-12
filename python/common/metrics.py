from dataclasses import dataclass
from typing import Literal

RegressionMetricName = Literal['mae', 'rmse', 'r2', 'mape']


@dataclass(frozen=True)
class RegressionMetrics:
    """Container for regression evaluation metrics.

    Attributes:
        mae: Mean Absolute Error.
        mse: Mean Squared Error.
        r2: R-squared score.
        mape: Mean Absolute Percentage Error.
    """

    mae: float
    rmse: float
    r2: float
    mape: float

    def __repr__(self) -> str:
        """Returns a compact string representation of the metric values."""

        mae_str = f'{self.mae:.3e}'.replace('e+', 'e')
        rmse_str = f'{self.rmse:.3e}'.replace('e+', 'e')
        r2_str = f'{self.r2:.3f}'
        mape_str = f'{self.mape * 100:.2f}%'

        return (
            f'Metrics(mae={mae_str}, mse={rmse_str}, '
            f'r2={r2_str}, mape={mape_str})'
        )

    def get_value(self, name: RegressionMetricName) -> float:
        match name:
            case 'mae':
                return self.mae
            case 'rmse':
                return self.rmse
            case 'r2':
                return self.r2
            case 'mape':
                return self.mape
            case _:
                raise ValueError(f'Unknown metric name: {name}')
