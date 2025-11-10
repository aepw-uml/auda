from typing import Any, Dict, List

import numpy as np
from typer import Argument, Option, Typer, echo

from auda.dat import get_task_spec, run_pipeline
from auda.utils.table import Table

app = Typer()


def parse_initial_inputs(initial_inputs_str: str) -> Dict[str, Any]:
    initial_inputs: Dict[str, Any] = {}
    for pair_str in initial_inputs_str.split(';'):
        if pair_str == '':
            continue

        if '=' not in pair_str:
            raise ValueError(f'Invalid initial input format: {pair_str}')

        key, value = pair_str.split('=', 1)
        initial_inputs[key] = value

    return initial_inputs


@app.command(name='check', help='Check if a pipeline is valid.')
def check(
    task_ids: List[str] = Argument(..., help='List of tasks in the pipeline.'),
    initial_inputs_str: str = Option(
        '', '-i', '--inputs', help='Initial inputs in key=value format.'
    ),
) -> None:
    if not task_ids:
        return echo('(Empty pipeline)')

    initial_inputs = parse_initial_inputs(initial_inputs_str)
    task_specs = [get_task_spec(task_id) for task_id in task_ids]

    context_keys = set(initial_inputs.keys())
    for i, task_spec in enumerate(task_specs):
        print(
            f'[{i + 1}] {task_spec.id} ({task_spec.kind})\n    {task_spec.description}'
        )
        missing_inputs = [
            input_key
            for input_key, input_spec in task_spec.input_specs.items()
            if input_key not in context_keys and input_spec.required
        ]

        if missing_inputs:
            echo('-' * 80)
            echo(f'Error: Task "{task_spec.id}" is missing inputs:')
            for input_key in missing_inputs:
                echo(f'  - {input_key}')
            return

        context_keys.update(task_spec.output_specs.keys())

    echo('-' * 80)
    echo('Pipeline is valid.')


@app.command(name='run', help='Run a pipeline consisting of specific tasks.')
def run(
    task_ids: List[str] = Argument(..., help='List of tasks in the pipeline.'),
    initial_inputs_str: str = Option(
        '', '-i', '--inputs', help='Initial inputs in key=value format.'
    ),
    format: str = Option('table', '-f', '--format', help='Either json or table.'),
    output_keys_str: str = Option(
        '', '-k', '--keys', help='Specific output key to display.'
    ),
) -> None:
    # Run the pipeline and get the outputs
    initial_inputs = parse_initial_inputs(initial_inputs_str)
    outputs, pipeline = run_pipeline(task_ids, initial_inputs)

    def stringify(value: Any) -> str:
        if isinstance(value, np.ndarray):
            return str(np.round(value, 2).tolist())

        return str(value)

    if output_keys_str:
        output_keys = output_keys_str.split(',')
        for output_key in output_keys:
            if output_key not in outputs:
                return echo(f'Output key "{output_key}" not found in outputs.')

            echo(f'<{output_key}>')
            echo(stringify(outputs[output_key]))
            echo('-' * 80)
    else:
        match format:
            case 'json':
                echo(outputs)
            case 'table':
                if not outputs:
                    return echo('(No outputs)')

                table = Table(['Key', 'Value'])
                for output_keys_str, value in outputs.items():
                    value_str = stringify(value)
                    if len(value_str) > 63:
                        value_str = value_str[0:63] + '...'
                    table.append_row([output_keys_str, value_str])

                echo(table)

    # Call the scheduled callbacks
    for callback in pipeline.callbacks:
        callback()
