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

    dataset, _ = dataset_cls().fetch(**AllowCustomArgs.parse_kwargs(ctx.args))

    match task:
        case 'NNForecasting':
            from experiment.nn_forecasting_experiment import (
                NNForecastingExperiment,
            )

            X, y = dataset.X, dataset.y
            assert y is not None, (
                'Target variable y is required for regression tasks.'
            )

            experiment = NNForecastingExperiment(
                name='NN Forecasting', description=''
            )
            experiment.setup(X=X, y=y, **context)
            experiment.run()
            print(experiment.get_metrics())
        case _:
            raise ValueError(f'Unknown task: {task}')
