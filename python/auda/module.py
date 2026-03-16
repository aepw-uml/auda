from dataset.dataset import Dataset, DatasetSchema
from typer import Context, Typer

app = Typer(name='module', help='Manage modules')


@app.command(
    name='run',
    help='Run a module',
    context_settings={
        'allow_extra_args': True,
        'ignore_unknown_options': True,
    },
)
def run_module(ctx: Context, module: str, dataset_name: str) -> None:
    from pathlib import Path

    from dataset.global_year_plastic_production import (
        GlobalYearPlasticsProduction,
    )
    from dataset.year_ppc import YearPPC
    from dataset.year_pwg import YearPWG
    from dataset.year_trc import YearTRC
    from module.projection import (
        run_projection_experiments,
        save_projection_experiment_results,
    )
    from module.reconstruction import (
        run_reconstruction_experiment_groups,
        run_reconstruction_experiments,
        save_metric_table,
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
        case 'year_ppc':
            location = context.get('location', 'Japan')
            dataset, schema = YearPPC().fetch(location)
        case 'global_year_plastics_production':
            dataset, schema = GlobalYearPlasticsProduction().fetch()
        case _:
            raise ValueError(f'Unknown dataset: {dataset_name}')

    match module:
        case 'projection':
            group = run_projection_experiments(dataset, schema, context)
            save_projection_experiment_results(group)
        case 'reconstruction':
            group = run_reconstruction_experiments(dataset, schema, context)
            save_reconstruction_experiment_results(group)
        case 'multiple_reconstruction':
            num_experiments = int(context.get('num_experiments', '8'))
            _, average_metrics = run_reconstruction_experiment_groups(
                num_experiments, dataset, schema, context
            )

            module_path = Path('results') / 'module' / 'multiple_reconstruction'
            save_metric_table(average_metrics, module_path)
        case _:
            raise ValueError(f'Unknown module: {module}')


def parse_kwargs(args: list[str]) -> dict[str, str]:
    context: dict[str, str] = {}
    for arg in args:
        if not arg.startswith('--'):
            raise ValueError(f'Invalid argument: {arg}')

        if '=' not in arg:
            key = arg[2:]
            context[key] = 'true'
        else:
            key, value = arg[2:].split('=', 1)
            context[key] = value

    return context
