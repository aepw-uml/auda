from dataset.dataset import Dataset, DatasetSchema
from typer import Context, Typer

app = Typer(name='module', help='Manage modules')


@app.command(
    name='run',
    help='Run a module',
    context_settings={
        'allow_extra_args': True,
        'ignore_unknown_options': False,
    },
)
def run_module(ctx: Context, module: str, dataset_name: str) -> None:
    from dataset.year_pwg import YearPWG
    from dataset.year_trc import YearTRC
    from module.projection import (
        get_projection_experiment_group,
        save_projection_experiment_results,
    )
    from module.reconstruction import (
        get_reconstruction_experiment_group,
        save_reconstruction_experiment_results,
    )

    context: dict[str, str] = parse_kwargs(ctx.args)
    dataset: Dataset | None
    schema: DatasetSchema | None

    match dataset_name:
        case 'year_trc':
            location = context.get('location', 'United States')
            dataset, schema = YearTRC().fetch(location)
        case 'year_pwg':
            location = context.get('location', 'United States')
            dataset, schema = YearPWG().fetch(location)
        case _:
            raise ValueError(f'Unknown dataset: {dataset_name}')

    match module:
        case 'projection':
            group = get_projection_experiment_group(context)
            group.set_context(**context)
            group.run(dataset, schema)
            save_projection_experiment_results(group)
        case 'reconstruction':
            group = get_reconstruction_experiment_group(context)
            group.set_context(**context)
            group.run(dataset, schema)
            save_reconstruction_experiment_results(group)


def parse_kwargs(args: list[str]) -> dict[str, str]:
    return dict(zip(args[::2], args[1::2]))
