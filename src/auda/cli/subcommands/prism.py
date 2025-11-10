from uuid import UUID

from typer import Argument, Option, Typer, echo
from typing_extensions import Annotated

from auda.core import project
from auda.models.prism_data_extraction import PrismDataExtraction
from auda.services.data import DataService
from auda.services.prism import PrismService
from auda.utils.table import Table

app = Typer()


@app.command(
    name='list-columns',
    help='List all data columns in the PRISM database.',
)
def display_data_columns_with_count(
    exclude_existing: bool = Option(
        False,
        '--exclude-existing',
        help='Exclude columns that already exist in the local database.',
    ),
    only_existing: bool = Option(
        False,
        '--only-existing',
        help='Only include columns that already exist in the local database.',
    ),
) -> None:
    prism_service = project.singleton_registry.get(PrismService)
    tuple_list = prism_service.get_data_points_with_description_and_counts()

    if exclude_existing or only_existing:
        data_service = project.singleton_registry.get(DataService)
        data_column_metadata = data_service.get_data_column_metadata()
        data_column_name_set = set(
            [row.original_column_name for row in data_column_metadata]
        )

        real_tuple_list = []
        for item in tuple_list:
            column_name = item[0]
            if exclude_existing and column_name not in data_column_name_set:
                real_tuple_list.append(item)
            elif only_existing and column_name in data_column_name_set:
                real_tuple_list.append(item)

        tuple_list = real_tuple_list

    table = Table(['Column Name', 'Description', 'Count'])
    for column_name, description, count in tuple_list:
        table.append_row([column_name, description, count])

    echo(table)


@app.command(
    name='list-documents',
    help='List the metadata of all documents in the PRISM database.',
)
def display_documents(csv_path: Annotated[str | None, Option('--csv')] = None) -> None:
    prism_service = project.singleton_registry.get(PrismService)
    documents = prism_service.get_all_documents()

    data = [
        [
            document.document_name,
            document.document_url,
            f'{document.file_size_mb:.2f}' if document.file_size_mb else '',
            str(document.created_at.strftime('%m/%d/%Y %H:%M:%S')),
        ]
        for document in documents
    ]

    columns = ['Document Name', 'URL', 'FILE SIZE (MB)', 'CREATED AT']
    if csv_path:
        import pandas as pd

        df = pd.DataFrame(data, columns=columns)  # type: ignore
        df.to_csv(csv_path, index=False)
    else:
        echo(Table(columns).append_rows(data))


@app.command(
    name='document-id-of',
    help='Get the document ID of a specific data point in data tables.',
)
def get_document(
    table_name: str = Argument(help='The table name of the data point.'),
    column_name: str = Argument(help='The column name of the data point.'),
    location: str = Argument(help='The location associated with the data point.'),
    year: str = Argument(help='The year associated with the data point.'),
    value: str = Argument(help='The value of the data point.'),
) -> None:
    data_service = project.singleton_registry.get(DataService)
    data_column_metadata = data_service.get_data_column_metadata()

    data_column_metadatum = next(
        (
            row
            for row in data_column_metadata
            if row.table_name == table_name and row.column_name == column_name
        ),
        None,
    )

    if data_column_metadatum is None:
        return echo(
            f'No metadata found for table "{table_name}" and column "{column_name}".'
        )

    # Get the location ID in the PRISM database
    prism_service = project.singleton_registry.get(PrismService)
    locations = prism_service.get_locations()
    prism_location = next(
        (_location for _location in locations if _location.location_name == location),
        None,
    )
    if prism_location is None:
        return echo(f'No location found with the name "{location}".')

    location_id = prism_location.id

    # Get the data point ID in the PRISM database
    original_column_name = data_column_metadatum.original_column_name
    data_points = prism_service.get_data_points()
    prism_data_point = next(
        (
            _data_point
            for _data_point in data_points
            if _data_point.name == original_column_name
        ),
        None,
    )
    if prism_data_point is None:
        return echo(
            f'No data point found with the name "{original_column_name}" in the PRISM '
            'database.'
        )
    data_point_id = prism_data_point.id

    # Get the data extraction record for the specified data point, location, year, and
    # value
    prim_data_extraction = prism_service.get_data_extraction(
        UUID(str(data_point_id)), UUID(str(location_id)), year, value
    )

    if len(prim_data_extraction) == 0:
        return echo(
            f'No data extraction record found for data point ID "{data_point_id}", '
            f'location ID "{location_id}", year "{year}", and value "{value}".'
        )
    elif len(prim_data_extraction) > 1:
        echo(
            f'Multiple data extraction records found for data point ID '
            f'"{data_point_id}", location ID "{location_id}", year "{year}", and '
            f'value "{value}".'
        )

    first_extraction: PrismDataExtraction = prim_data_extraction[0]
    print(first_extraction.file_id)


@app.command(
    name='document-metadata',
    help='Display metadata of a specific document in the PRISM database.',
)
def display_document_metadata(
    document_id: str = Argument(help='The ID of the document to display metadata for.'),
) -> None:
    prism_service = project.singleton_registry.get(PrismService)
    prism_document = prism_service.get_document_by_id(UUID(document_id))

    if prism_document is None:
        return echo(f'No document found with ID "{document_id}".')

    data = [
        ['Document Name', prism_document.document_name],
        ['URL', prism_document.document_url],
        [
            'File Size (MB)',
            f'{prism_document.file_size_mb:.2f}' if prism_document.file_size_mb else '',
        ],
        ['Created At', str(prism_document.created_at.strftime('%m/%d/%Y %H:%M:%S'))],
    ]

    echo(Table(['Field', 'Value']).append_rows(data))
