from dataset.year_pw import YearPW
from experiment.reconstruction import ReconstructionExperiment
from step.model.drift_baseline import DriftBaseline
from step.model.exponential_smoothing import ExponentialSmoothing
from step.model.naive_persistence import NaivePersistence
from step.model.polynomial_regression import PolynomialRegression
from step.model.ridge_regression import RidgeRegression

if __name__ == '__main__':
    dataset, schema = YearPW().fetch('Japan')
    X, y = dataset.X, dataset.y

    if y is None:
        raise ValueError(
            'Label data is required for reconstruction experiment.'
        )

    # ==========================================================================

    experiment = ReconstructionExperiment(
        name='Reconstruction (Naive Persistence)',
        description=(
            'Reconstruct the original time series with naive persistence.'
        ),
        regressor=NaivePersistence,
    )

    experiment.setup(X, y)
    experiment.run()
    experiment.logger.info(experiment.get_metrics())
    experiment.finish()

    # ==========================================================================

    experiment = ReconstructionExperiment(
        name='Reconstruction (Drift Baseline)',
        description=(
            'Reconstruct the original time series with drift baseline.'
        ),
        regressor=DriftBaseline,
    )
    experiment.setup(X, y)
    experiment.run()
    experiment.logger.info(experiment.get_metrics())
    experiment.finish()

    # ==========================================================================

    experiment = ReconstructionExperiment(
        name='Reconstruction (Exponential Smoothing)',
        description=(
            'Reconstruct the original time series with exponential smoothing.'
        ),
        regressor=ExponentialSmoothing,
    )
    experiment.setup(X, y, use_scaler=False)
    experiment.run()
    experiment.logger.info(experiment.get_metrics())
    experiment.finish()

    # ==========================================================================

    experiment = ReconstructionExperiment(
        name='Reconstruction (Polynomial Regression)',
        description=(
            'Reconstruct the original time series with polynomial regression.'
        ),
        regressor=PolynomialRegression,
    )
    experiment.setup(X, y)
    experiment.run()
    experiment.logger.info(experiment.get_metrics())
    experiment.finish()

    # ==========================================================================

    experiment = ReconstructionExperiment(
        name='Reconstruction (Ridge Regression)',
        description=(
            'Reconstruct the original time series with ridge regression.'
        ),
        regressor=RidgeRegression,
    )
    experiment.setup(X, y)
    experiment.run()
    experiment.logger.info(experiment.get_metrics())
    experiment.finish()

    # ==========================================================================
