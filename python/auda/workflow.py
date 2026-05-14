from dataset import dataset_map
from typer import Context, Typer

from auda.common import AllowCustomArgs

app = Typer(name='workflow', help='Manage workflows')


@app.command(
    name='run',
    help='Run a workflow',
    context_settings=AllowCustomArgs.context_settings,
)
def run_workflow(ctx: Context, dataset_name: str, workflow: str) -> None:
    context: dict[str, str] = AllowCustomArgs.parse_kwargs(ctx.args)
    dataset_cls = dataset_map.get(dataset_name)
    if not dataset_cls:
        print(f"Dataset '{dataset_name}' not found.")
        return

    dataset, schema = dataset_cls().fetch(
        **AllowCustomArgs.parse_kwargs(ctx.args)
    )

    match workflow:
        case 'Reconstruction':
            from workflow.reconstruction_workflow import (
                ReconstructionWorkflow,
            )

            ReconstructionWorkflow().run(dataset, schema, **context)
        case 'MultipleReconstruction':
            from workflow.multiple_reconstruction_workflow import (
                MultipleReconstructionWorkflow,
            )

            MultipleReconstructionWorkflow().run(dataset, schema, **context)
        case 'SEToleranceCoefficientSweep':
            from workflow.se_tolerance_coefficient_sweep_workflow import (
                SEToleranceCoefficientSweepWorkflow,
            )

            SEToleranceCoefficientSweepWorkflow().run(
                dataset, schema, **context
            )
        case 'ComplexityOrderingRobustness':
            from workflow.complexity_ordering_robustness_workflow import (
                ComplexityOrderingRobustnessWorkflow,
            )

            ComplexityOrderingRobustnessWorkflow().run(
                dataset, schema, **context
            )
        case 'Forecasting':
            from workflow.forecasting_workflow import ForecastingWorkflow

            ForecastingWorkflow().run(dataset, schema, **context)
        case 'MultipleForecasting':
            from workflow.multiple_forecasting_workflow import (
                MultipleForecastingWorkflow,
            )

            MultipleForecastingWorkflow().run(dataset, schema, **context)
        case 'Correlation':
            from workflow.correlation_workflow import CorrelationWorkflow

            CorrelationWorkflow().run(dataset, schema, **context)
        case 'FeatureImportances':
            from workflow.feature_importances_workflow import (
                FeatureImportancesflow,
            )

            FeatureImportancesflow().run(dataset, schema, **context)
        case 'NNForecasting':
            from workflow.nn_forecasting_workflow import NNForecastingWorkflow

            NNForecastingWorkflow().run(dataset, schema, **context)
        case 'MultipleNNForecasting':
            from workflow.multiple_nn_forecasting_workflow import (
                MultipleNNForecastingWorkflow,
            )

            MultipleNNForecastingWorkflow().run(dataset, schema, **context)
        case 'MultivariateForecasting':
            from workflow.multivariate_forecasting_workflow import (
                MultivariateForecastingWorkflow,
            )

            MultivariateForecastingWorkflow().run(dataset, schema, **context)
        case 'MultipleMultivariateForecasting':
            from workflow.multiple_multivariate_forecasting_workflow import (
                MultipleMultivariateForecastingWorkflow,
            )

            MultipleMultivariateForecastingWorkflow().run(
                dataset, schema, **context
            )
        case _:
            raise ValueError(f'Unknown workflow: {workflow}')
