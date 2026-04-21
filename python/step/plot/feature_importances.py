from typing import override

import numpy as np
from step.plot.plotter import Plotter


class FeatureImportancesPlotter(Plotter):
    @override
    def plot(self, feature_importances: np.ndarray) -> None:
        pass
