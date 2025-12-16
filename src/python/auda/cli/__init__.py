from typer import Typer

from .pipe import app as pipe_app

app = Typer()

app.add_typer(pipe_app)
