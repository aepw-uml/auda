import logging
from typing import Annotated

import typer
from typer import Typer

from auda.dataset import app as dataset_app
from auda.migration import app as migration_app
from auda.workflow import app as workflow_app

from .state import State

app = Typer(rich_markup_mode=None)


@app.callback()
def main(
    verbose: Annotated[bool, typer.Option('--verbose', '-v')] = False,
) -> None:
    if verbose:
        State.verbose = True
        logging.getLogger().setLevel(logging.DEBUG)


app.add_typer(migration_app)
app.add_typer(dataset_app)
app.add_typer(workflow_app)
