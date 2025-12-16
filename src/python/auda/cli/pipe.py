from enum import Enum
from typing import Any, Dict, List, Tuple

from auda.step import *
from auda.utils.pipeline import (
    IOSpec,
    create_pipeline,
    get_all_step_specs,
    get_kind,
)
from auda.utils.table import Table
from auda.utils.types import format_type
from typer import Argument, Option, Typer, echo

app = Typer(name='pipe', help='Pipeline related commands.')

STEP_STR_DELIMITER = ':'
INPUTS_STR_DELIMITER = ';'
INPUTS_STR_KEY_VALUE_DELIMITER = '='
KEYS_STR_DELIMITER = ','


@app.command(name='list', help='Display all step specs.')
def list_spec_specs(
    kind: str = Option('', '--kind', '-k', help='Step kind.'),
) -> None:
    table = Table(['ID', 'Kind', 'Description'])
    for step_spec in get_all_step_specs():
        step_spec_kind = get_kind(step_spec)
        if kind and step_spec != kind:
            continue

        table.append_row(step_spec.id, step_spec_kind, step_spec.description)

    echo(table)


@app.command(
    name='info', help='Display specific information about a step spec.'
)
def check_info(
    spec_spec_id: str = Argument(help='The ID of the spec spec.'),
) -> None:
    spec_spec_id = spec_spec_id.upper()
    spec_specs = get_all_step_specs()
    step_spec = next(
        (spec for spec in spec_specs if spec.id == spec_spec_id), None
    )
    if not step_spec:
        return echo(f'Step spec with ID {spec_spec_id} not found.')

    id = step_spec.id
    kind = get_kind(step_spec)
    description = step_spec.description
    implementation = step_spec.implementation
    input_spec_map = step_spec.input_spec_map
    output_spec_map = step_spec.output_spec_map

    echo(f'ID: {id}')
    echo(f'Kind: {kind}')
    echo(f'Description: {description}')
    echo(
        f'Implementation: {implementation.__module__}.{implementation.__name__}'
    )

    def echo_spec_map(spec_map: Dict[str, IOSpec]) -> None:
        if not spec_map:
            echo(' ' * 4 + '(none)')
        for name, spec in input_spec_map.items():
            name = name.lower()
            type_str = format_type(spec.dtype)
            extra = '' if spec.required else f', default={spec.default}'
            required = f' (required: {spec.required}{extra})'

            echo(f'{" " * 4}- {name}: {type_str}{required}')

    echo('\nInput Specs:')
    echo_spec_map(input_spec_map)

    echo('\nOutput Specs:')
    echo_spec_map(output_spec_map)


@app.command(name='run', help='Run a pipeline consisting of specified steps.')
def run(
    step_strs: List[str] = Argument(..., help='List of steps in the pipeline.'),
    format: str = Option(
        'table', '-f', '--format', help='Output format (table|json).'
    ),
    output_names_str: str = Option(
        '', '-n', '--names', help='Output keys to display.'
    ),
    lowercase_names: bool = Option(
        True,
        '-l',
        '--lowercase-names',
        help='Convert output keys to lowercase.',
    ),
):
    def process_step_str(step_str: str) -> Tuple[str, Dict[Enum | str, Any]]:
        inputs: Dict[Enum | str, Any] = {}
        if STEP_STR_DELIMITER in step_str:
            step_id, inputs_str = step_str.split(STEP_STR_DELIMITER, 1)
            inputs_strs = inputs_str.split(INPUTS_STR_DELIMITER)

            raw_inputs = {}
            for input in inputs_strs:
                sp = input.split(INPUTS_STR_KEY_VALUE_DELIMITER, 1)
                raw_inputs[sp[0].upper()] = sp[1]

            inputs.update(raw_inputs)
        else:
            step_id = step_str

        return step_id, inputs

    step_ids: List[str] = []
    step_inputs: List[Dict[Enum | str, str]] = []
    for step_str in step_strs:
        step_id, inputs = process_step_str(step_str)
        step_ids.append(step_id)
        step_inputs.append(inputs)

    step_ids = [step_id.upper() for step_id in step_ids]
    pipeline = create_pipeline(step_ids)
    pipeline.run(step_inputs=step_inputs)

    names = (
        [name.upper() for name in output_names_str.split(KEYS_STR_DELIMITER)]
        if output_names_str != ''
        else []
    )
    outputs = (
        pipeline.context
        if names == []
        else {
            name: pipeline.context.get(name)
            for name in names
            if name in pipeline.context.keys()
        }
    )

    # Convert output keys to lowercase if specified
    if lowercase_names:
        outputs = {
            name.lower(): value for name, value in pipeline.context.items()
        }

    # Print outputs in the specified format
    match format:
        case 'json':
            import json

            echo(json.dumps(outputs, indent=2))
        case 'table':
            if outputs == {}:
                return

            table = Table()
            for name, value in outputs.items():
                table.append_row(name, str(value))

            echo(table)
        case _:
            echo(f'Unsupported format: {format}')
