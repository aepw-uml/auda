from auda.dat.datasets.__common import DatasetOSName


class StatISName:
    SAMPLES = DatasetOSName.SAMPLES


class StatOSName:
    # Basic statistics
    NUM_SAMPLES = 'num_samples'
    NUM_FEATURES = 'num_features'
    X_MEAN = 'x_mean'
    X_MINIMUM = 'x_minimum'
    X_MAXIMUM = 'x_maximum'
    X_STANDARD_DEVIATION = 'x_standard_deviation'
    X_VARIANCE = 'x_variance'
    Y_MEAN = 'y_mean'
    Y_MINIMUM = 'y_minimum'
    Y_MAXIMUM = 'y_maximum'
    Y_STANDARD_DEVIATION = 'y_standard_deviation'
    Y_VARIANCE = 'y_variance'


STAT_KIND = 'stat'
