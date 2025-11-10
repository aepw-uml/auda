from typer import Typer

from auda.core import project
from auda.services.data import DataService

app = Typer()


@app.command(
    name='create-data-tables',
    help='Create data tables based on the `data_column_metadata` table.',
)
def create_data_tables() -> None:
    data_service = project.singleton_registry.get(DataService)
    data_service.create_data_tables()


@app.command(
    name='drop-data-tables',
    help='Drop data tables based on the `data_column_metadata` table.',
)
def drop_data_tables() -> None:
    data_service = project.singleton_registry.get(DataService)
    data_service.drop_data_tables()
