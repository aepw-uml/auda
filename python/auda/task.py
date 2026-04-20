from dataset import dataset_map
from typer import Context, Typer

from auda.common import AllowCustomArgs

app = Typer(name='task', help='Manage tasks')


@app.command(
    name='run',
    help='Run a task',
    context_settings=AllowCustomArgs.context_settings,
)
def run_task(ctx: Context, dataset_name: str, task: str) -> None:
    context: dict[str, str] = AllowCustomArgs.parse_kwargs(ctx.args)
    dataset_cls = dataset_map.get(dataset_name)
    if not dataset_cls:
        print(f"Dataset '{dataset_name}' not found.")
        return

    dataset, schema = dataset_cls().fetch(
        **AllowCustomArgs.parse_kwargs(ctx.args)
    )

    match task:
        case 'Reconstruction':
            from workflow.reconstruction_workflow import (
                ReconstructionWorkflow,
            )

            ReconstructionWorkflow().run(dataset, schema, **context)
        case 'MultipleReconstruction':
            from workflow.multiple_reconstruction_workflow import (
                MultipleReconstructionWorkflow,
            )

            MultipleReconstructionWorkflow().run(
                dataset, schema, **context
            )
        case 'Forecasting':
            from workflow.forecasting_workflow import ForecastingWorkflow

            ForecastingWorkflow().run(dataset, schema, **context)
        case 'NNForecasting':
            from workflow.nn_forecasting_workflow import NNForecastingWorkflow

            NNForecastingWorkflow().run(dataset, schema, **context)
        case _:
            raise ValueError(f'Unknown task: {task}')
