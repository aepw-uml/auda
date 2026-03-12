from typing import Any, Literal, Self

from tabulate import tabulate


class Table:
    def __init__(
        self,
        headers: list[str] | None = None,
        bold_headers: bool = True,
        fmt: Literal['simple', 'plain', 'github'] = 'github',
        colalign: list[str] | None = None,
    ) -> None:
        """Initializes a Table instance.

        Attributes:
            tables: A list of lists representing the table data.
            headers: A list of strings representing the column headers.
            bold_headers: A boolean indicating whether to bold the headers.
            fmt: A string representing the table format in tabulate.
        """

        self.headers: list[str] | None = headers
        self.bold_headers: bool = bold_headers
        self.fmt: Literal['simple', 'plain', 'github'] = fmt
        self.colalign: list[str] | None = colalign

        self.table: list[list[str]] = []

    def append_row(self, *items: Any) -> Self:
        """Adds a row to the table.

        Args:
            items: A list of values representing a row in the table.
        """

        if self.headers and len(items) != len(self.headers):
            raise ValueError(
                f'Row length {len(items)} does not match header length '
                f'{len(self.headers)}.'
            )

        self.table.append([str(item) for item in items])

        return self

    def __repr__(self) -> str:
        """Converts a table (list of lists) to a string representation.

        This function uses the `tabulate` library to format the table in a
        readable way.
        """

        headers = self.headers
        if self.headers and self.bold_headers:
            headers = [f'\033[1m{header}\033[0m' for header in self.headers]

        return tabulate(
            self.table,
            headers=headers or (),
            tablefmt=self.fmt,
            colalign=self.colalign,
            disable_numparse=True,
        )
