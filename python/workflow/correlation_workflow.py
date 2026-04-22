from pathlib import Path
from typing import override

import numpy as np
from common.dataset import Dataset, DatasetSchema
from common.workflow.workflow import Workflow
from step.feature.correlation import calculate_correlation_matrix
from step.plot.correlation_matrix import CorrelationMatrixPlotter


class CorrelationWorkflow(Workflow):
    @override
    def run(self, dataset: Dataset, schema: DatasetSchema, **context) -> None:
        super().run(**context)

        # Calculate correlation matrix
        correlation_matrix = calculate_correlation_matrix(dataset.X)
        print(correlation_matrix)

        # Save the correlation matrix to a file
        dir_path = Path('results') / 'correlation' / 'correlation_matrix'
        dir_path.mkdir(parents=True, exist_ok=True)
        np.savetxt(
            dir_path / 'correlation_matrix.csv',
            correlation_matrix,
            delimiter=',',
        )

        # Plot correlation matrix and save it to a file
        plotter = CorrelationMatrixPlotter(schema, '')
        plotter.plot(correlation_matrix)

        file_path = plotter.save(dir_path)
        print(f'Saved correlation matrix plot to "{file_path}".')
