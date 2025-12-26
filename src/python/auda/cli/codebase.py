from typer import Typer

app = Typer(name='cache', help='Cache related commands.')


@app.command(name='collect', help='')
def collect_codebase():
    pass
