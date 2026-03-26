import logging
from typing import Annotated

import typer
from typer import Typer

from auda.migration import app as migration_app
from auda.module import app as module_app

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
app.add_typer(module_app)
