from typing import override

from common.dataset import Dataset, DatasetSchema
from common.task import Task
from experiment.projection_experiment import (
    ProjectionExperimentGroup,
    getARIMAProjection,
    getDriftBaselineProjection,
    getExponentialSmoothingProjection,
    getGaussianProcessProjection,
    getNaivePersistenceProjection,
    getRidgeRegressionProjection,
    getSupportVectorRegressionProjection,
    getTheilSenProjection,
)


class ProjectionTask(Task):
    @override
    def run(self, dataset: Dataset, schema: DatasetSchema, **context) -> None:
        group = ProjectionExperimentGroup(name='Projection Experiments')
        group.set_context(**context)
        group.add(getNaivePersistenceProjection(**context))
        group.add(getDriftBaselineProjection(**context))
        group.add(getExponentialSmoothingProjection(**context))
        group.add(getRidgeRegressionProjection(**context))
        group.add(getGaussianProcessProjection(**context))
        group.add(getSupportVectorRegressionProjection(**context))
        group.add(getTheilSenProjection(**context))
        group.add(getARIMAProjection(**context))

        group.run(dataset, schema)

        for experiment in group.experiments:
            experiment.get_metrics()
