from sqlalchemy import Integer
from sqlalchemy.orm import Mapped, mapped_column

from .base_model import SABase


class DataColumnMetadata(SABase):
    """
    Metadata for columns of data tables, linking them to their corresponding attributes
    in a spreadsheet.

    Args:
        id: Unique identifier.
        table_name: The name of the table.
        column_name: The name of the column in the table.
        original_column_name: The original name of the column in the spreadsheet.
        data_type: The data type of the column (e.g., integer, float, string, etc.).
        unit: The unit of measurement for the column (if applicable).
        description: A brief description of the column.
    """

    __tablename__ = 'data_column_metadata'

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    table_name: Mapped[str] = mapped_column(nullable=False)
    column_name: Mapped[str] = mapped_column(nullable=False)
    original_column_name: Mapped[str] = mapped_column(nullable=False)
    data_type: Mapped[str] = mapped_column(nullable=False)
    unit: Mapped[str] = mapped_column(nullable=True)
    description: Mapped[str] = mapped_column(nullable=True)
