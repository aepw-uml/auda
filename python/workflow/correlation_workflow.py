from pathlib import Path
from typing import override

from common.dataset import Dataset, DatasetSchema
from common.workflow.workflow import Workflow
from step.feature.correlation import calculate_correlation_matrix
from step.plot.correlation_matrix import CorrelationMatrixPlotter


class CorrelationWorkflow(Workflow):
    @override
    def run(self, dataset: Dataset, schema: DatasetSchema) -> None:
        # Calculate correlation matrix
        correlation_matrix = calculate_correlation_matrix(dataset.X)
        print(correlation_matrix)

        # Plot correlation matrix and save it to a file
        plotter = CorrelationMatrixPlotter(schema, '')
        plotter.plot(correlation_matrix)

        dir_path = Path('results') / 'correlation_matrix'
        file_path = plotter.save(dir_path)
        print(f'Saved correlation matrix plot to "{file_path}".')
