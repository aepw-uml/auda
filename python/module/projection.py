from pathlib import Path
from typing import cast, override

from common.files import save_content_to_file
from common.hyperparameters import get_hyperparameters_str
from common.metrics import RegressionMetrics
from common.names import to_kebab
from dataset.dataset import Dataset, DatasetSchema
from dataset.year_trc import YearTRC
from experiment.experiment_group import ExperimentGroup
from experiment.projection import ProjectionExperiment
from step.model.drift_baseline import DriftBaseline
from step.model.exponential_smoothing import ExponentialSmoothing
from step.model.gaussian_process_regression import GaussianProcessRegression
from step.model.naive_persistence import NaivePersistence
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


def getNaivePersistenceProjection(**_) -> ProjectionExperiment:
    return ProjectionExperiment(
        name='Naive Persistence',
        description=(
            'Project the original time series with naive persistence.'
        ),
        regressor=NaivePersistence,
    )


def getDriftBaselineProjection(**_) -> ProjectionExperiment:
    return ProjectionExperiment(
        name='Drift Baseline',
        description=('Project the original time series with drift baseline.'),
        regressor=DriftBaseline,
    )


def getExponentialSmoothingProjection(**_) -> ProjectionExperiment:
    experiment = ProjectionExperiment(
        name='Exponential Smoothing',
        description=(
            'Project the original time series with exponential smoothing.'
        ),
        regressor=ExponentialSmoothing,
    )

    experiment.set_context(use_scaler=False)

    return experiment


def getPolynomialRegressionProjection(**_) -> ProjectionExperiment:
    experiment = ProjectionExperiment(
        name='Polynomial Regression',
        description=(
            'Project the original time series with polynomial regression.'
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


def getRidgeRegressionProjection(**_) -> ProjectionExperiment:
    experiment = ProjectionExperiment(
        name='Ridge Regression',
        description=('Project the original time series with ridge regression.'),
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


def getGaussianProcessProjection(**_) -> ProjectionExperiment:
    experiment = ProjectionExperiment(
        name='Gaussian Process Regression',
        description=(
            'Project the original time series with Gaussian process regression.'
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


def getSupportVectorRegressionProjection(
    **context: str,
) -> ProjectionExperiment:
    experiment = ProjectionExperiment(
        name='Support Vector Regression',
        description=(
            'Project the original time series with support vector regression.'
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


class ProjectionExperimentGroup(ExperimentGroup):
    @override
    def run(self, dataset: Dataset, schema: DatasetSchema) -> None:
        super().run(dataset)

        X, y = dataset.X, dataset.y
        assert y is not None

        seed = int(self.context.get('seed', 42))
        metric = self.context.get('metric', 'mape')
        for experiment in self.experiments:
            experiment = cast(ProjectionExperiment, experiment)

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
        X, y = dataset.X, dataset.y

        if y is None:
            raise ValueError(
                'Label data is required for projection experiment.'
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


def get_projection_experiment_group(
    context: dict[str, str],
) -> ProjectionExperimentGroup:
    group = ProjectionExperimentGroup(name='Projection Experiments')
    group.add_experiment(getNaivePersistenceProjection(**context))
    group.add_experiment(getDriftBaselineProjection(**context))
    group.add_experiment(getExponentialSmoothingProjection(**context))
    group.add_experiment(getPolynomialRegressionProjection(**context))
    group.add_experiment(getRidgeRegressionProjection(**context))
    group.add_experiment(getGaussianProcessProjection(**context))
    group.add_experiment(getSupportVectorRegressionProjection(**context))

    return group


def run_projection_experiments(
    dataset: Dataset,
    schema: DatasetSchema,
    context: dict[str, str] | None = None,
) -> None:
    if context is None:
        context = {}

    experiment_group = get_projection_experiment_group(context)
    experiment_group.set_context(**context)
    experiment_group.run(dataset, schema)


def save_projection_experiment_results(
    group: ProjectionExperimentGroup,
) -> None:
    # Module path to save the results of the projection experiments.
    module_path = Path('results') / 'module' / 'projection'

    # Build a metric table and save it to a file.
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

    # Build a hyperparameter table and save it to a file.
    hyperparameter_table = Table(headers=['Experiment', 'Hyperparameters'])
    for experiment in group.experiments:
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
    print()
    print(hyperparameter_table.__repr__())
    print()

    # Save the plots of each experiment.
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
    # dataset, schema = YearPWG().fetch('Japan')
    # dataset, schema = YearPWG().fetch('United States')
    dataset, schema = YearTRC().fetch('United States')

    # We find that tuning the gamma hyperparameter of SVR drops the performance
    # instead.
    run_projection_experiments(
        dataset,
        schema,
        {'metric': 'mape', 'svr_tune_gamma': 'True', 'seed': '151'},
    )
