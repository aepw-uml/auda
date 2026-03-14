from pathlib import Path

from common.files import save_content_to_file
from common.hyperparameters import get_hyperparameters_str
from common.metrics import RegressionMetricName, RegressionMetrics
from common.names import to_kebab
from dataset.dataset import Dataset, DatasetSchema
from dataset.year_pwg import YearPWG
from dataset.year_trc import YearTRC
from experiment.reconstruction import ReconstructionExperiment
from step.model.cubic_spline import CubicSpline
from step.model.gaussian_process_regression import GaussianProcessRegression
from step.model.linear_interpolation import LinearInterpolation
from step.model.moving_average_interpolation import MovingAverageInterpolation
from step.model.polynomial_regression import PolynomialRegression
from step.model.ridge_regression import RidgeRegression
from step.model.support_vector_regression import SupportVectorRegression
from step.plot.gaussian_process_regression import (
    GaussianProcessRegressionPlotter,
)
from step.plot.plotter import Plotter
from step.plot.polynomial_regression import PolynomialRegressionPlotter
from step.plot.ridge_regression import RidgeRegressionPlotter
from step.plot.support_vector_regression import (
    SupportVectorRegressionPlotter,
)
from util.table import Table


def getLinearInterpolationReconstruction() -> ReconstructionExperiment:
    """Builds a linear-interpolation reconstruction experiment."""

    experiment = ReconstructionExperiment(
        name='Linear Interpolation',
        description=(
            'Reconstruct the original time series with linear interpolation.'
        ),
        regressor=LinearInterpolation,
    )

    return experiment


def getMovingAverageInterpolationReconstruction() -> ReconstructionExperiment:
    """Builds a moving-average-interpolation reconstruction experiment."""

    experiment = ReconstructionExperiment(
        name='Moving Average Interpolation',
        description=(
            'Reconstruct the original time series with moving average '
            'interpolation.'
        ),
        regressor=MovingAverageInterpolation,
    )

    return experiment


def getCubicSplineInterpolationReconstruction() -> ReconstructionExperiment:
    """Builds a cubic-spline-interpolation reconstruction experiment."""

    experiment = ReconstructionExperiment(
        name='Cubic Spline Interpolation',
        description=(
            'Reconstruct the original time series with cubic spline '
            'interpolation.'
        ),
        regressor=CubicSpline,
    )

    return experiment


def getPolynomialRegressionReconstruction() -> ReconstructionExperiment:
    """Builds a polynomial-regression reconstruction experiment."""

    experiment = ReconstructionExperiment(
        name='Polynomial Regression',
        description=(
            'Reconstruct the original time series with polynomial regression.'
        ),
        regressor=PolynomialRegression,
    )

    experiment.set_context(
        plotter_factory=PolynomialRegressionPlotter,
        tuning_parameters={
            'hyperparameter_names': ['degree'],
            'search_space': [[(2.0, 9.0)]],
            'elite_fractions': [0.12, 0.06],
            'refinement_widths': [[2], [0.5]],
            'sampling_scales': ['uniform'],
        },
    )

    return experiment


def getRidgeRegressionReconstruction() -> ReconstructionExperiment:
    """Builds a ridge-regression reconstruction experiment."""

    experiment = ReconstructionExperiment(
        name='Ridge Regression',
        description=(
            'Reconstruct the original time series with ridge regression.'
        ),
        regressor=RidgeRegression,
    )

    experiment.set_context(
        plotter_factory=RidgeRegressionPlotter,
        tuning_parameters={
            'hyperparameter_names': ['degree', 'alpha'],
            'search_space': [[(2.0, 9.0)], [(1e-6, 1e3)]],
            'elite_fractions': [0.12, 0.06],
            'refinement_widths': [[2, 0.5], [0.5, 0.1]],
            'sampling_scales': ['uniform', 'log_uniform'],
        },
    )

    return experiment


def getGaussianProcessReconstruction() -> ReconstructionExperiment:
    """Builds a Gaussian-process reconstruction experiment."""

    experiment = ReconstructionExperiment(
        name='Gaussian Process Regression',
        description=(
            'Reconstruct the original time series with Gaussian process '
            'regression.'
        ),
        regressor=GaussianProcessRegression,
    )

    experiment.set_context(
        plotter_factory=GaussianProcessRegressionPlotter,
        tuning_parameters={
            'hyperparameter_names': ['length_scale', 'noise_level'],
            'search_space': [[(1e-3, 1e3)], [(1e-6, 1e1)]],
            'elite_fractions': [0.12, 0.06],
            'refinement_widths': [[10.0, 0.5], [2.0, 0.1]],
            'sampling_scales': ['log_uniform', 'log_uniform'],
        },
    )

    return experiment


def getSupportVectorRegressionReconstruction(
    tune_gamma: bool = True,
) -> ReconstructionExperiment:
    """Builds a support-vector-regression reconstruction experiment."""

    experiment = ReconstructionExperiment(
        name='Support Vector Regression',
        description=(
            'Reconstruct the original time series with support vector '
            'regression.'
        ),
        regressor=SupportVectorRegression,
    )

    experiment.set_context(
        plotter_factory=SupportVectorRegressionPlotter,
    )

    if tune_gamma:
        experiment.set_context(
            tuning_parameters={
                'hyperparameter_names': ['C', 'epsilon', 'gamma'],
                'search_space': [
                    [(0.1, 100.0)],
                    [(0.001, 1.0)],
                    [(0.001, 1.0)],
                ],
                'elite_fractions': [0.12, 0.06],
                'refinement_widths': [
                    [10.0, 0.2, 0.1],
                    [2.0, 0.05, 0.02],
                ],
                'sampling_scales': [
                    'log_uniform',
                    'log_uniform',
                    'log_uniform',
                ],
            },
        )
    else:
        experiment.set_context(
            tuning_parameters={
                'hyperparameter_names': ['C', 'epsilon'],
                'search_space': [
                    [(0.1, 100.0)],
                    [(0.001, 1.0)],
                ],
                'elite_fractions': [0.12, 0.06],
                'refinement_widths': [
                    [10.0, 0.2],
                    [2.0, 0.05],
                ],
                'sampling_scales': ['log_uniform', 'log_uniform'],
            },
        )

    return experiment


def run_reconstruction_experiments(
    dataset: Dataset,
    schema: DatasetSchema,
    plot_title: str = '',
    metric: RegressionMetricName = 'mape',
    svr_tune_gamma: bool = True,
    seed: int = 42,
) -> None:
    """Runs the reconstruction experiments and saves their artifacts.

    Args:
        dataset: Dataset containing the feature matrix and target vector.
        schema: Schema describing the dataset columns and units.
        plot_title: Optional title to use for generated plots.
        metric: Metric used during hyperparameter tuning.
        svr_tune_gamma: Whether to tune the gamma hyperparameter for SVR.
    """

    X, y = dataset.X, dataset.y

    if y is None:
        raise ValueError(
            'Label data is required for reconstruction experiments.'
        )

    if X.shape[0] != y.shape[0]:
        raise ValueError(
            'Number of samples in features and labels must be the same.'
        )

    if y.ndim != 1:
        raise ValueError('Labels must be one-dimensional.')

    experiments: list[ReconstructionExperiment] = [
        getLinearInterpolationReconstruction(),
        getMovingAverageInterpolationReconstruction(),
        getCubicSplineInterpolationReconstruction(),
        getPolynomialRegressionReconstruction(),
        getRidgeRegressionReconstruction(),
        getGaussianProcessReconstruction(),
        getSupportVectorRegressionReconstruction(tune_gamma=svr_tune_gamma),
    ]

    for experiment in experiments:
        # Set the random seed for reproducibility.
        experiment.seed = seed

        # Inject schema into the experiment context.
        experiment.context['schema'] = schema

        # Inject metric into the experiment context for hyperparameter tuning if
        # tuning parameters are defined.
        if 'tuning_parameters' in experiment.context:
            experiment.context['tuning_parameters']['metric'] = metric

        experiment.setup(X, y)
        experiment.run()
        experiment.logger.info(experiment.get_metrics())
        experiment.finish()

    module_path = Path('results') / 'module' / 'reconstruction'

    metric_table = Table(headers=['Experiment', 'MAE', 'RMSE', 'R²', 'MAPE'])
    for experiment in experiments:
        # Set the random seed for reproducibility.
        experiment.seed = seed

        metrics: RegressionMetrics = experiment.get_metrics()
        [mae_str, rmse_str, r2_str, mape_str] = metrics.item_strs()
        metric_table.append_row(
            experiment.name,
            mae_str,
            rmse_str,
            r2_str,
            mape_str,
        )

    metric_table_path: Path = module_path / 'metric_table'
    save_content_to_file(metric_table_path, metric_table.__repr__())
    print(f'Saved metric table to "{metric_table_path}".')
    print()
    print(metric_table.__repr__())
    print()

    hyperparameter_table = Table(headers=['Experiment', 'Hyperparameters'])
    for experiment in experiments:
        hyperparameters_str = get_hyperparameters_str(
            experiment.hyperparameters
        )
        hyperparameter_table.append_row(
            experiment.name,
            hyperparameters_str if hyperparameters_str else '-',
        )

    hyperparameter_table_path: Path = module_path / 'hyperparameter_table'
    save_content_to_file(
        hyperparameter_table_path,
        hyperparameter_table.__repr__(),
    )
    print(f'Saved hyperparameter table to "{hyperparameter_table_path}".')
    print()
    print(hyperparameter_table.__repr__())
    print()

    if X.shape[1] != 1:
        print(
            'Skipping plot export for reconstruction experiments because '
            'plotting requires exactly one feature column.'
        )
        return

    plots_dir: Path = module_path / 'plots'
    for experiment in experiments:
        experiment.context['plot_title'] = plot_title

        plotter: Plotter | None = experiment.plot()
        if plotter is None:
            continue

        plot_path: Path = plots_dir / to_kebab(experiment.name)
        file_path: str = plotter.save(plot_path)
        print(f'Saved plot for "{experiment.name}" to "{file_path}".')


if __name__ == '__main__':
    location = 'United Kingdom'
    dataset, schema = YearPWG().fetch('United States')
    dataset, schema = YearPWG().fetch('Japan')
    dataset, schema = YearTRC().fetch('United States')

    run_reconstruction_experiments(
        dataset, schema, metric='mape', svr_tune_gamma=True, seed=150
    )
