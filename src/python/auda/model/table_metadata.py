from sqlalchemy import Integer, String
from sqlalchemy.orm import Mapped, mapped_column

from .base_model import SABase


class TableMetadata(SABase):
    """
    Represents a table in the database.

    Args:
        id: Unique identifier for the table.
        name: Unique name of the table.
        type: Type of the table, which can be either 'system' or 'data'. System
            tables are used for internal purposes, while data tables are used to
            store processed AEPW data.
    """

    __tablename__ = 'table_metadata'

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    name: Mapped[str] = mapped_column(primary_key=True, index=True)
    type: Mapped[str] = mapped_column(String, nullable=False)
