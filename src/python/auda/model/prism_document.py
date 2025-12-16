from datetime import datetime
from typing import Optional

from sqlalchemy import DOUBLE_PRECISION, TEXT, TIMESTAMP, UUID, VARCHAR, Text
from sqlalchemy.orm import Mapped, mapped_column

from .base_model import SABase


class PrismDocument(SABase):
    __tablename__ = 'documents'

    id: Mapped[UUID] = mapped_column(UUID, primary_key=True, index=True)
    document_name: Mapped[str] = mapped_column(Text, nullable=False)
    document_url: Mapped[Optional[str]] = mapped_column(VARCHAR(255))
    file_size_mb: Mapped[Optional[float]] = mapped_column(DOUBLE_PRECISION)
    file_type: Mapped[Optional[str]] = mapped_column(VARCHAR(50))

    document_status: Mapped[str] = mapped_column(VARCHAR(50))
    extraction_status: Mapped[str] = mapped_column(VARCHAR(50))
    reported_year: Mapped[str] = mapped_column(VARCHAR(4))
    publishers: Mapped[str] = mapped_column(TEXT)
    organization_type: Mapped[str] = mapped_column(VARCHAR(100))
    organization_country: Mapped[str] = mapped_column(VARCHAR(100))

    created_at: Mapped[datetime] = mapped_column(
        TIMESTAMP(timezone=True), nullable=False
    )
    created_by: Mapped[UUID] = mapped_column(UUID(as_uuid=True))
    updated_at: Mapped[datetime] = mapped_column(
        TIMESTAMP(timezone=True), nullable=False
    )
    updated_by: Mapped[UUID] = mapped_column(UUID(as_uuid=True))
