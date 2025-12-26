from typer import Typer

from .cache import app as cache_app
from .codebase import app as codebase_app
from .pipe import app as pipe_app

app = Typer()

app.add_typer(pipe_app)
app.add_typer(cache_app)
app.add_typer(codebase_app)
