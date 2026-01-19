from shutil import rmtree

from auda.core import auda
from typer import Typer, echo

app = Typer(name='cache', help='Manages cache')


@app.command(name='clear', help='Clear the application cache.')
def clear_cache():
    rmtree(auda.cache_dir)
    auda.cache_dir.mkdir(parents=True, exist_ok=True)

    echo(f'Cleared cache at {auda.cache_dir}')
