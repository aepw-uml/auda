from typer import Typer

from auda.module import app as module_app

app = Typer(rich_markup_mode=None)
app.add_typer(module_app)
