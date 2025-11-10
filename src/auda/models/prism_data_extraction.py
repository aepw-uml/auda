from datetime import datetime

from sqlalchemy import TIMESTAMP, UUID, Boolean, Integer, Text
from sqlalchemy.orm import Mapped, mapped_column

from .base_model import SABase


class PrismDataExtraction(SABase):
    __tablename__ = 'data_extractions'

    id: Mapped[UUID] = mapped_column(UUID, primary_key=True, index=True)
    file_id: Mapped[UUID] = mapped_column(UUID, nullable=False)
    data_point_id: Mapped[UUID] = mapped_column(UUID, nullable=False)
    location_id: Mapped[UUID] = mapped_column(UUID, nullable=False)
    year: Mapped[int] = mapped_column(Integer, nullable=False)
    value: Mapped[str] = mapped_column(Text, nullable=False)
    unit: Mapped[str] = mapped_column(Text, nullable=True)
    source_page: Mapped[int] = mapped_column(Integer, nullable=True)
    source_context: Mapped[str] = mapped_column(Text, nullable=True)
    data_source_score: Mapped[int] = mapped_column(Integer, nullable=True)
    data_completeness_group: Mapped[str] = mapped_column(Text, nullable=True)
    data_completeness_completeness_assessment: Mapped[str] = mapped_column(
        Text, nullable=True
    )
    data_completeness_score: Mapped[int] = mapped_column(Integer, nullable=True)
    geographic_level_type: Mapped[str] = mapped_column(Text, nullable=True)
    geographic_level_score: Mapped[int] = mapped_column(Integer, nullable=True)
    temporal_relevance_year_difference: Mapped[int] = mapped_column(
        Integer, nullable=True
    )
    temporal_relevance_score: Mapped[int] = mapped_column(Integer, nullable=True)
    data_type: Mapped[str] = mapped_column(Text, nullable=True)
    data_type_score: Mapped[int] = mapped_column(Integer, nullable=True)
    inconsistency_violated_logic: Mapped[str] = mapped_column(Text, nullable=True)
    inconsistency_score: Mapped[int] = mapped_column(Integer, nullable=True)
    overall_quality_score: Mapped[int] = mapped_column(Integer, nullable=True)
    validation_status: Mapped[str] = mapped_column(Text, nullable=True)
    is_duplicated: Mapped[bool] = mapped_column(Boolean, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        TIMESTAMP(timezone=True), nullable=False
    )
    created_by: Mapped[UUID] = mapped_column(UUID, nullable=True)
    validated_at: Mapped[datetime] = mapped_column(
        TIMESTAMP(timezone=True), nullable=True
    )
    validated_by: Mapped[UUID] = mapped_column(UUID, nullable=True)
