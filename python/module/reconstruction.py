from pathlib import Path
from typing import cast, override

from common.files import save_content_to_file
from common.hyperparameters import get_hyperparameters_str
from common.metrics import RegressionMetrics
from common.names import to_kebab
from dataset.dataset import Dataset, DatasetSchema
from dataset.year_pwg import YearPWG
from dataset.year_trc import YearTRC
from experiment.experiment_group import ExperimentGroup
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


def getLinearInterpolationReconstruction(**_) -> ReconstructionExperiment:
    """Builds a linear-interpolation reconstruction experiment."""

    experiment = ReconstructionExperiment(
        name='Linear Interpolation',
        description=(
            'Reconstruct the original time series with linear interpolation.'
        ),
        regressor=LinearInterpolation,
    )

    return experiment


def getMovingAverageInterpolationReconstruction(
    **_,
) -> ReconstructionExperiment:
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


def getCubicSplineInterpolationReconstruction(**_) -> ReconstructionExperiment:
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


def getPolynomialRegressionReconstruction(**_) -> ReconstructionExperiment:
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
            'elite_fractions': [0.3, 0.3],
            'refinement_widths': [[3], [1.0]],
            'sampling_scales': ['uniform'],
        },
    )

    return experiment


def getRidgeRegressionReconstruction(**_) -> ReconstructionExperiment:
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
            'elite_fractions': [0.3, 0.3],
            'refinement_widths': [[3, 1.0], [1.0, 0.2]],
            'sampling_scales': ['uniform', 'log_uniform'],
        },
    )

    return experiment


def getGaussianProcessReconstruction(**_) -> ReconstructionExperiment:
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
            'elite_fractions': [0.3, 0.3],
            'refinement_widths': [[12.0, 1.0], [3.0, 0.25]],
            'sampling_scales': ['log_uniform', 'log_uniform'],
        },
    )

    return experiment


def getSupportVectorRegressionReconstruction(
    **context,
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

    if context.get('svr_tune_gamma', 'False').lower() == 'true':
        experiment.set_context(
            tuning_parameters={
                'hyperparameter_names': ['C', 'epsilon', 'gamma'],
                'search_space': [
                    [(0.1, 100.0)],
                    [(0.001, 1.0)],
                    [(0.001, 1.0)],
                ],
                'elite_fractions': [0.3, 0.3],
                'refinement_widths': [
                    [12.0, 0.35, 0.2],
                    [3.0, 0.1, 0.05],
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
                'elite_fractions': [0.3, 0.3],
                'refinement_widths': [
                    [12.0, 0.35],
                    [3.0, 0.1],
                ],
                'sampling_scales': ['log_uniform', 'log_uniform'],
            },
        )

    return experiment


class ReconstructionExperimentGroup(ExperimentGroup):
    """Runs the configured reconstruction experiments."""

    @override
    def run(self, dataset: Dataset, schema: DatasetSchema) -> None:
        """Runs the experiments after injecting shared context.

        Args:
            dataset: Dataset containing the feature matrix and target vector.
            schema: Schema describing the dataset columns and units.
        """

        super().run(dataset)

        X, y = dataset.X, dataset.y
        assert y is not None

        seed = int(self.context.get('seed', 42))
        self.logger.info(f'Using seed {seed}.')

        metric = self.context.get('metric', 'mape')
        self.logger.info(f'Using metric "{metric}" for hyperparameter tuning.')

        for experiment in self.experiments:
            experiment = cast(ReconstructionExperiment, experiment)

            # Set the random seed for reproducibility.
            experiment.seed = seed

            # Inject schema into the experiment context.
            experiment.context['schema'] = schema

            # Inject metric into the experiment context for hyperparameter
            # tuning if tuning parameters are defined.
            if 'tuning_parameters' in experiment.context:
                experiment.context['tuning_parameters']['metric'] = metric

            experiment.setup(X, y)
            experiment.run()
            experiment.logger.info(experiment.get_metrics())
            experiment.finish()

    @override
    def check_dataset(self, dataset: Dataset) -> None:
        """Validates the dataset required by reconstruction experiments.

        Args:
            dataset: Dataset containing the feature matrix and target vector.

        Raises:
            ValueError: If the dataset is incompatible with reconstruction.
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


def get_reconstruction_experiment_group(
    context: dict[str, str],
) -> ReconstructionExperimentGroup:
    """Builds the reconstruction experiment group.

    Args:
        tune_gamma: Whether to tune the gamma hyperparameter for SVR.

    Returns:
        The configured reconstruction experiment group.
    """

    group = ReconstructionExperimentGroup(name='Reconstruction Experiments')
    group.add_experiment(getLinearInterpolationReconstruction(**context))
    group.add_experiment(getMovingAverageInterpolationReconstruction(**context))
    group.add_experiment(getCubicSplineInterpolationReconstruction(**context))
    group.add_experiment(getPolynomialRegressionReconstruction(**context))
    group.add_experiment(getRidgeRegressionReconstruction(**context))
    group.add_experiment(getGaussianProcessReconstruction(**context))
    group.add_experiment(getSupportVectorRegressionReconstruction(**context))

    return group


def run_reconstruction_experiments(
    dataset: Dataset,
    schema: DatasetSchema,
    context: dict[str, str] | None = None,
) -> ReconstructionExperimentGroup:
    """Runs the reconstruction experiments and saves their artifacts.

    Args:
        dataset: Dataset containing the feature matrix and target vector.
        schema: Schema describing the dataset columns and units.
        context: Optional dictionary containing context values for the
            experiments

    Returns:
        The reconstruction experiment group containing the run experiments.
    """

    if context is None:
        context = {}

    group = get_reconstruction_experiment_group(context)
    group.set_context(**context)
    group.run(dataset, schema)

    return group


def save_reconstruction_experiment_results(
    group: ReconstructionExperimentGroup,
) -> None:
    module_path = Path('results') / 'module' / 'reconstruction'

    metric_table = Table(headers=['Experiment', 'MAE', 'RMSE', 'R²', 'MAPE'])
    for experiment in group.experiments:
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
    for experiment in group.experiments:
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

    plots_dir: Path = module_path / 'plots'
    for experiment in group.experiments:
        experiment.context['plot_title'] = ''
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
        dataset,
        schema,
        {'metric': 'mape', 'svr_tune_gamma': 'False'},
    )
