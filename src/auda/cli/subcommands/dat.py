from typing import Optional

from typer import Argument, Option, Typer, echo

from auda.utils.pipeline import get_all_specs
from auda.utils.table import Table
from auda.utils.types import format_type

app = Typer()


@app.command(name='list', help='Display all available task specs.')
def list_task_specs(
    kind: Optional[str] = Option(None, '--kind', '-k', help='Filter by task kind'),
) -> None:
    table = Table(['ID', 'Kind', 'Description'])
    for spec in get_all_specs():
        if kind and spec.kind != kind:
            continue

        table.append_row([spec.id, spec.kind, spec.description or ''])

    echo(table)


@app.command(
    name='info', help='Display detailed information about a specific task spec.'
)
def check_info(task_spec_id: str = Argument(help='The ID of the task spec.')) -> None:
    task_specs = get_all_specs()
    task_spec = next((spec for spec in task_specs if spec.id == task_spec_id), None)
    if not task_spec:
        return echo(f'Task spec with ID {task_spec_id} not found.')

    id = task_spec.id
    kind = task_spec.kind
    description = task_spec.description or ''
    implementation = task_spec.implementation
    input_specs = task_spec.input_specs
    output_specs = task_spec.output_specs

    echo(f'ID: {id}')
    echo(f'Kind: {kind}')
    echo(f'Description: {description}')
    echo(f'Implementation: {implementation.__module__}.{implementation.__name__}')

    echo('\nInput Specs:')
    if not input_specs:
        echo('  (none)')
    for name, spec in input_specs.items():
        extra = '' if spec.required else f', default={spec.default}'
        echo(
            f'  - {name}: {format_type(spec.dtype)} (required: {spec.required}{extra})'
        )

    echo('\nOutput Specs:')
    if not output_specs:
        echo('  (none)')
    for name, spec in output_specs.items():
        echo(f'  - {name}: {format_type(spec.dtype)}')
