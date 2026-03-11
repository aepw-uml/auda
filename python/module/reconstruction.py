from dataset.year_pw import YearPW
from experiment.reconstruction import Reconstruction
from sklearn.linear_model import LinearRegression
from step.model.naive_persistence import NaivePersistence

if __name__ == '__main__':
    dataset, schema = YearPW().fetch('Japan')
    X, y = dataset.X, dataset.y

    if y is None:
        raise ValueError(
            'Label data is required for reconstruction experiment.'
        )

    experiment = Reconstruction(
        name='Reconstruction',
        description='Reconstruct the original time series.',
        regressor=NaivePersistence,
    )

    experiment.setup(X, y)
    experiment.run()
    print(experiment.get_metrics())

    experiment = Reconstruction(
        name='Reconstruction',
        description='Reconstruct the original time series.',
        regressor=NaivePersistence,
    )

    experiment = Reconstruction(
        name='Reconstruction',
        description='Reconstruct the original time series.',
        regressor=LinearRegression,
    )
    experiment.setup(X, y)
    experiment.run()
    print(experiment.get_metrics())
