import os

# Configure Matplotlib before workflow modules import pyplot.
os.environ.setdefault('MPLCONFIGDIR', '/tmp/auda-matplotlib')
os.environ.setdefault('XDG_CACHE_HOME', '/tmp/auda-cache')

import matplotlib

matplotlib.use('Agg')

from .complexity_ordering_robustness_workflow import (
    ComplexityOrderingRobustnessWorkflow,
)
from .forecasting_workflow import ForecastingWorkflow
from .multiple_multivariate_forecasting_workflow import (
    MultipleMultivariateForecastingWorkflow,
)
from .multiple_reconstruction_workflow import MultipleReconstructionWorkflow
from .multivariate_forecasting_workflow import MultivariateForecastingWorkflow
from .nn_forecasting_workflow import NNForecastingWorkflow
from .reconstruction_workflow import ReconstructionWorkflow
from .se_tolerance_coefficient_sweep_workflow import (
    SEToleranceCoefficientSweepWorkflow,
)

__all__ = [
    'ComplexityOrderingRobustnessWorkflow',
    'ForecastingWorkflow',
    'MultipleMultivariateForecastingWorkflow',
    'MultipleReconstructionWorkflow',
    'MultivariateForecastingWorkflow',
    'NNForecastingWorkflow',
    'ReconstructionWorkflow',
    'SEToleranceCoefficientSweepWorkflow',
]
