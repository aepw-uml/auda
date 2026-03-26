from datetime import datetime

from sqlalchemy import TIMESTAMP, UUID, Integer, String
from sqlalchemy.orm import Mapped, mapped_column

from .base_model import SABase


class PrismLocation(SABase):
    __tablename__ = 'locations'

    id: Mapped[UUID] = mapped_column(UUID, primary_key=True, index=True)
    location_code: Mapped[str] = mapped_column(String(50), nullable=False)
    location_name: Mapped[str] = mapped_column(String(255), nullable=False)
    type: Mapped[str] = mapped_column(String(50), nullable=False)
    aepw_country_type: Mapped[int] = mapped_column(Integer, nullable=False)
    parent_id: Mapped[UUID] = mapped_column(UUID, nullable=True)
    created_at: Mapped[datetime] = mapped_column(TIMESTAMP, nullable=False)
