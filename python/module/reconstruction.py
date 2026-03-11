from dataset.year_pw import YearPW
from experiment.reconstruction import Reconstruction
from step.model.naive_persistence import NaivePersistence
from step.model.polynomial_regression import PolynomialRegression

if __name__ == '__main__':
    dataset, schema = YearPW().fetch('Japan')
    X, y = dataset.X, dataset.y

    if y is None:
        raise ValueError(
            'Label data is required for reconstruction experiment.'
        )

    experiment = Reconstruction(
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

    experiment = Reconstruction(
        name='Reconstruction (Linear Polynomial Regression)',
        description=(
            'Reconstruct the original time series with polynomial regression.'
        ),
        regressor=PolynomialRegression,
    )
    experiment.setup(X, y, hyperparameters={'degree': 3})
    experiment.run()
    experiment.logger.info(experiment.get_metrics())
    experiment.finish()
