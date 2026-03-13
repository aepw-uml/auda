from pathlib import Path

from common.files import save_content_to_file
from common.hyperparameters import get_hyperparameters_str
from common.metrics import RegressionMetricName, RegressionMetrics
from dataset.dataset import Dataset, DatasetSchema
from dataset.year_pw import YearPW
from experiment.reconstruction import ReconstructionExperiment
from step.model.drift_baseline import DriftBaseline
from step.model.exponential_smoothing import ExponentialSmoothing
from step.model.gaussian_process_regression import GaussianProcessRegression
from step.model.naive_persistence import NaivePersistence
from step.model.polynomial_regression import PolynomialRegression
from step.model.ridge_regression import RidgeRegression
from step.model.support_vector_regression import SupportVectorRegression
from step.plot.plotter import RegressionPlotter
from util.table import Table


def getNaivePersistenceReconstruction() -> ReconstructionExperiment:
    return ReconstructionExperiment(
        name='Naive Persistence',
        description=(
            'Reconstruct the original time series with naive persistence.'
        ),
        regressor=NaivePersistence,
    )


def getDriftBaselineReconstruction() -> ReconstructionExperiment:
    return ReconstructionExperiment(
        name='Drift Baseline',
        description=(
            'Reconstruct the original time series with drift baseline.'
        ),
        regressor=DriftBaseline,
    )


def getExponentialSmoothingReconstruction() -> ReconstructionExperiment:
    experiment = ReconstructionExperiment(
        name='Exponential Smoothing',
        description=(
            'Reconstruct the original time series with exponential smoothing.'
        ),
        regressor=ExponentialSmoothing,
    )

    experiment.set_context(use_scaler=False)

    return experiment


def getPolynomialRegressionReconstruction() -> ReconstructionExperiment:
    experiment = ReconstructionExperiment(
        name='Polynomial Regression',
        description=(
            'Reconstruct the original time series with polynomial regression.'
        ),
        regressor=PolynomialRegression,
    )

    experiment.set_context(
        plotter_factory=RegressionPlotter,
        tuning_parameters={
            'hyperparameter_names': ['degree'],
            'search_space': [[(2, 12)]],
            'elite_fractions': [0.12, 0.06],
            'refinement_widths': [[2], [0.5]],
        },
    )

    return experiment


def getRidgeRegressionReconstruction() -> ReconstructionExperiment:
    experiment = ReconstructionExperiment(
        name='Ridge Regression',
        description=(
            'Reconstruct the original time series with ridge regression.'
        ),
        regressor=RidgeRegression,
    )

    experiment.set_context(
        tuning_parameters={
            'hyperparameter_names': ['alpha'],
            'search_space': [[(1e-6, 1e3)]],
            'elite_fractions': [0.12, 0.06],
            'refinement_widths': [[0.5], [0.1]],
        }
    )

    return experiment


def getGaussianProcessReconstruction() -> ReconstructionExperiment:
    experiment = ReconstructionExperiment(
        name='Gaussian Process Regression',
        description=(
            'Reconstruct the original time series with Gaussian process '
            'regression.'
        ),
        regressor=GaussianProcessRegression,
    )

    experiment.set_context(
        tuning_parameters={
            'hyperparameter_names': ['length_scale', 'noise_level'],
            'search_space': [[(1e-3, 1e3)], [(1e-6, 1e1)]],
            'elite_fractions': [0.12, 0.06],
            'refinement_widths': [[10.0, 0.5], [2.0, 0.1]],
        }
    )

    return experiment


def getSupportVectorRegressionReconstruction(
    tune_gamma: bool = True,
) -> ReconstructionExperiment:
    experiment = ReconstructionExperiment(
        name='Support Vector Regression',
        description=(
            'Reconstruct the original time series with support vector '
            'regression.'
        ),
        regressor=SupportVectorRegression,
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
            }
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
            }
        )

    return experiment


def run_reconstruction_experiments(
    dataset: Dataset,
    schema: DatasetSchema,
    metric: RegressionMetricName = 'mape',
    svr_tune_gamma: bool = True,
) -> None:
    X, y = dataset.X, dataset.y

    if y is None:
        raise ValueError(
            'Label data is required for reconstruction experiment.'
        )

    if X.shape[0] != y.shape[0]:
        raise ValueError(
            'Number of samples in features and labels must be the same.'
        )

    if X.shape[1] != 1 or y.ndim != 1:
        raise ValueError(
            'Features must have exactly one column and labels must be '
            'one-dimensional.'
        )

    experiments: list[ReconstructionExperiment] = [
        getNaivePersistenceReconstruction(),
        getDriftBaselineReconstruction(),
        getExponentialSmoothingReconstruction(),
        getPolynomialRegressionReconstruction(),
        getRidgeRegressionReconstruction(),
        getGaussianProcessReconstruction(),
        getSupportVectorRegressionReconstruction(tune_gamma=svr_tune_gamma),
    ]

    for experiment in experiments:
        # Inject schema into the experiment context.
        experiment.context['schema'] = schema

        # Inject metric into the experiment context for hyperparameter tuning if
        # tuning parameters are defined.
        if 'tuning_parameters' in experiment.context:
            experiment.context['tuning_parameters']['metric'] = metric

        experiment.setup(X, y)
        experiment.run()
        experiment.logger.info(experiment.get_metrics())

        plotter = experiment.plot()
        if plotter is not None:
            plotter.show()

        experiment.finish()

    # Module path to save the results of the reconstruction experiments.
    module_path = Path('results') / 'module' / 'reconstruction'

    # Build a metric table and save it to a file.
    metric_table = Table(headers=['Experiment', 'MAE', 'RMSE', 'R²', 'MAPE'])
    for experiment in experiments:
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

    # Build a hyperparameter table and save it to a file.
    hyperparameter_table = Table(headers=['Experiment', 'Hyperparameters'])
    for experiment in experiments:
        hyerparameters_str = get_hyperparameters_str(experiment.hyperparameters)
        hyperparameter_table.append_row(
            experiment.name,
            hyerparameters_str if hyerparameters_str else '-',
        )
    hyperparameter_table_path: Path = module_path / 'hyperparameter_table'
    save_content_to_file(
        hyperparameter_table_path,
        hyperparameter_table.__repr__(),
    )
    print(f'Saved hyperparameter table to "{hyperparameter_table_path}".')


if __name__ == '__main__':
    # location = 'United States'
    location = 'Japan'
    dataset, schema = YearPW().fetch(location)
    run_reconstruction_experiments(dataset, schema, metric='r2')
