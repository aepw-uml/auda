from typer import Typer

from auda.migration import app as migration_app
from auda.module import app as module_app

app = Typer(rich_markup_mode=None)
app.add_typer(migration_app)
app.add_typer(module_app)
