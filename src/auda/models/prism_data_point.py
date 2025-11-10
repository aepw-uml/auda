from datetime import datetime

from sqlalchemy import TIMESTAMP, UUID, String, Text
from sqlalchemy.orm import Mapped, mapped_column

from .base_model import SABase


class PrismDataPoint(SABase):
    """
    Metadata for individual data points used in the PRISM system.

    Args:
        id: Unique UUID identifier generated via gen_random_uuid().
        name: The name of the data point.
        description: A text description of the data point.
        data_group: The group or category the data point belongs to.
        created_at: Timestamp of when the data point was created.
    """

    __tablename__ = 'data_points'

    id: Mapped[UUID] = mapped_column(UUID, primary_key=True, index=True)
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    description: Mapped[str] = mapped_column(Text, nullable=True)
    data_group: Mapped[str] = mapped_column(String(100), nullable=False)
    created_at: Mapped[datetime] = mapped_column(TIMESTAMP, nullable=False)
