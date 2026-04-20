from dataset import dataset_map
from typer import Context, Typer

from auda.common import AllowCustomArgs

app = Typer(name='dataset', help='Manage datasets')


@app.command(
    name='list',
    help='List all datasets in the system.',
)
def list() -> None:
    for dataset_name in dataset_map.keys():
        print(dataset_name)


@app.command(
    name='show',
    help='Show details of a specific dataset.',
    context_settings=AllowCustomArgs.context_settings,
)
def show(ctx: Context, dataset_name: str) -> None:
    dataset_cls = dataset_map.get(dataset_name)
    if not dataset_cls:
        print(f"Dataset '{dataset_name}' not found.")
        return

    custom_args = AllowCustomArgs.parse_kwargs(ctx.args)
    dataset, schema = dataset_cls().fetch(**custom_args)
    print(schema)
    print(f'Number of samples: {len(dataset.X)}')
    print(dataset)
