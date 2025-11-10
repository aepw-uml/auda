from typer import Typer

from .subcommands.dat import app as app_dat
from .subcommands.llm import app as app_llm
from .subcommands.migration import app as app_migration
from .subcommands.pipe import app as app_pipe
from .subcommands.prism import app as app_prism
from .subcommands.table import app as app_table

app = Typer()

# Register all subcommands
app.add_typer(app_migration, name='migration')
app.add_typer(app_table, name='table')
app.add_typer(app_prism, name='prism')
app.add_typer(app_dat, name='dat')
app.add_typer(app_pipe, name='pipe')
app.add_typer(app_llm, name='llm')

__all__ = ['app']
