from pathlib import Path
from typing import override

from common.dataset import Dataset, DatasetSchema
from common.experiment.persistence import (
    save_hyperparameter_table,
    save_metric_table,
    save_plots,
    save_time_table,
)
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

        # Task path to save the results of the projection experiments.
        task_path = Path('results') / 'projection'

        # Save the metric table for the projection experiments.
        save_metric_table(group, task_path)

        # Save a hyperparameter table and save it to a file.
        save_hyperparameter_table(group, task_path)

        # Save the plots of each experiment.
        save_plots(group, task_path)

        # Save time.
        save_time_table(group, task_path)
