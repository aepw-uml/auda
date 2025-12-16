from typing import Any, List, Self

from tabulate import tabulate


class Table:
    def __init__(
        self,
        headers: List[str] | None = None,
        bold_headers: bool = True,
        fmt: str = 'plain',
    ) -> None:
        """Initializes a Table instance.

        Attributes:
            tables: A list of lists representing the table data.
            headers: A list of strings representing the column headers.
            bold_headers: A boolean indicating whether to bold the headers.
            fmt: A string representing the table format in tabulate.
        """
        self.table: List[List[str]] = []
        self.headers = headers
        self.bold_headers = bold_headers
        self.fmt = fmt

    def append_row(self, *row: Any) -> Self:
        """Adds a row to the table.

        Args:
            row: A list of values representing a row in the table.
        """
        if self.headers and len(row) != len(self.headers):
            raise ValueError(
                f'Row length {len(row)} does not match header length '
                f'{len(self.headers)}.'
            )

        self.table.append([str(item) for item in row])

        return self

    def append_rows(self, rows: List[List[Any]]) -> Self:
        """Adds multiple rows to the table.

        Args:
            rows: A list of lists, where each inner list represents a row in
                the table.
        """
        for row in rows:
            self.append_row(row)

        return self

    def __repr__(self) -> str:
        """Converts a table (list of lists) to a string representation.

        This function uses the `tabulate` library to format the table in a
        readable way.
        """
        headers = self.headers
        if self.headers and self.bold_headers:
            headers = [f'\033[1m{header}\033[0m' for header in self.headers]

        return tabulate(self.table, headers=headers or (), tablefmt=self.fmt)
