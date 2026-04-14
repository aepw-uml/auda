from typing import cast, override

from common.dataset import Dataset, DatasetSchema
from common.experiment.experiment_group import ExperimentGroup
from common.experiment.forecasting_experiment import ForecastingExperiment
from step.model.arima_regression import ARIMARegression
from step.model.drift_baseline import DriftBaseline
from step.model.exponential_smoothing import ExponentialSmoothing
from step.model.gaussian_process_regression import GaussianProcessRegression
from step.model.naive_persistence import NaivePersistence
from step.model.polynomial_regression import PolynomialRegression
from step.model.ridge_regression import RidgeRegression
from step.model.support_vector_regression import SupportVectorRegression
from step.model.theil_sen_regression import TheilSenRegression
from step.plot.arima_regression import ARIMARegressionPlotter
from step.plot.gaussian_process_regression import (
    GaussianProcessRegressionPlotter,
)
from step.plot.polynomial_regression import PolynomialRegressionPlotter
from step.plot.ridge_regression import RidgeRegressionPlotter
from step.plot.support_vector_regression import SupportVectorRegressionPlotter
from step.plot.theil_sen_regression import TheilSenRegressionPlotter


def getNaivePersistenceForecasting(**_) -> ForecastingExperiment:
    return ForecastingExperiment(
        name='Naive Persistence',
        description=(
            'Forecast the original time series with naive persistence.'
        ),
        regressor_cls=NaivePersistence,
    )


def getDriftBaselineForecasting(**_) -> ForecastingExperiment:
    return ForecastingExperiment(
        name='Drift Baseline',
        description=('Forecast the original time series with drift baseline.'),
        regressor_cls=DriftBaseline,
    )


def getExponentialSmoothingForecasting(**_) -> ForecastingExperiment:
    experiment = ForecastingExperiment(
        name='Exponential Smoothing',
        description=(
            'Forecast the original time series with exponential smoothing.'
        ),
        regressor_cls=ExponentialSmoothing,
    )
    experiment.set_context(use_scaler=False, use_target_scaler=False)
    return experiment


def getARIMAForecasting(**context) -> ForecastingExperiment:
    experiment = ForecastingExperiment(
        name='ARIMA Regression',
        description=('Forecast the original time series with ARIMA.'),
        regressor_cls=ARIMARegression,
    )

    experiment.set_context(
        **context,
        use_scaler=False,
        use_target_scaler=False,
        plotter_factory=ARIMARegressionPlotter,
    )
    experiment.hyperparameters = {
        'replace': True,
        'p': 2,
        'd': 0,
        'q': 1,
        'trend': 'n',
    }

    return experiment


def getTheilSenForecasting(**_) -> ForecastingExperiment:
    experiment = ForecastingExperiment(
        name='Theil-Sen Regression',
        description=(
            'Forecast the original time series with robust Theil-Sen '
            'regression.'
        ),
        regressor_cls=TheilSenRegression,
    )

    experiment.set_context(
        plotter_factory=TheilSenRegressionPlotter,
    )
    experiment.hyperparameters = {
        'replace': True,
        'window_size': 7,
    }

    return experiment


def getPolynomialRegressionForecasting(**_) -> ForecastingExperiment:
    experiment = ForecastingExperiment(
        name='Polynomial Regression',
        description=(
            'Forecast the original time series with polynomial regression.'
        ),
        regressor_cls=PolynomialRegression,
    )

    experiment.set_context(
        plotter_factory=PolynomialRegressionPlotter,
        tuning_parameters={
            'search_type': 'grid',
            'hyperparameter_names': ['degree'],
            'search_space': [[(2.0, 9.0)]],
            'sampling_scales': ['uniform'],
            'num_points_per_interval': 8,
        },
    )

    return experiment


def getRidgeRegressionForecasting(**_) -> ForecastingExperiment:
    experiment = ForecastingExperiment(
        name='Ridge Regression',
        description=(
            'Forecast the original time series with ridge regression.'
        ),
        regressor_cls=RidgeRegression,
    )

    experiment.set_context(
        plotter_factory=RidgeRegressionPlotter,
        tuning_parameters={
            'search_type': 'grid',
            'hyperparameter_names': ['degree', 'alpha'],
            'search_space': [[(2.0, 9.0)], [(1e-6, 1e3)]],
            'sampling_scales': ['uniform', 'log_uniform'],
            'num_points_per_interval': 8,
        },
    )

    return experiment


def getGaussianProcessForecasting(**_) -> ForecastingExperiment:
    experiment = ForecastingExperiment(
        name='Gaussian Process Regression',
        description=(
            'Forecast the original time series with Gaussian process '
            'regression.'
        ),
        regressor_cls=GaussianProcessRegression,
    )

    experiment.set_context(
        plotter_factory=GaussianProcessRegressionPlotter,
        tuning_parameters={
            'search_type': 'grid',
            'hyperparameter_names': ['length_scale', 'noise_level'],
            'search_space': [[(1e-3, 1e3)], [(1e-6, 1e1)]],
            'sampling_scales': ['log_uniform', 'log_uniform'],
        },
    )

    return experiment


def getSupportVectorRegressionForecasting(
    **context: str,
) -> ForecastingExperiment:
    experiment = ForecastingExperiment(
        name='Support Vector Regression',
        description=(
            'Forecast the original time series with support vector regression.'
        ),
        regressor_cls=SupportVectorRegression,
    )

    if bool(context.get('svr_tune_gamma', 0)):
        tuning_parameters = {
            'search_type': 'grid',
            'hyperparameter_names': ['C', 'epsilon', 'gamma'],
            'search_space': [
                [(0.1, 100.0)],
                [(0.001, 1.0)],
                [(0.001, 1.0)],
            ],
            'sampling_scales': [
                'log_uniform',
                'log_uniform',
                'log_uniform',
            ],
        }
    else:
        tuning_parameters = {
            'search_type': 'grid',
            'hyperparameter_names': ['C', 'epsilon'],
            'search_space': [
                [(0.1, 100.0)],
                [(0.001, 1.0)],
            ],
            'sampling_scales': [
                'log_uniform',
                'log_uniform',
            ],
        }

    experiment.set_context(
        plotter_factory=SupportVectorRegressionPlotter,
        tuning_parameters=tuning_parameters,
        **context,
    )

    return experiment


class ForecastingExperimentGroup(ExperimentGroup):
    @override
    def run(self, dataset: Dataset, schema: DatasetSchema) -> None:
        super().run(dataset)

        X, y = dataset.X, dataset.y
        assert y is not None

        seed = int(self.context.get('seed', 42))
        self.logger.info(f'Using seed {seed}.')

        metric = self.context.get('metric', 'mape')
        self.logger.info(f'Using metric "{metric}" for hyperparameter tuning.')

        for experiment in self.experiments:
            experiment = cast(ForecastingExperiment, experiment)

            # Set the random seed for reproducibility.
            experiment.seed = seed

            # Inject schema into the experiment context.
            experiment.context['schema'] = schema

            # Inject metric into the experiment context for hyperparameter
            # tuning if tuning parameters are defined.
            if 'tuning_parameters' in experiment.context:
                experiment.context['tuning_parameters']['metric'] = metric

            experiment.setup(X, y, **self.context)
            experiment.run()
            experiment.logger.info(experiment.get_metrics())

    @override
    def check_dataset(self, dataset: Dataset) -> None:
        X, y = dataset.X, dataset.y

        if y is None:
            raise ValueError(
                'Label data is required for forecasting experiment.'
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
