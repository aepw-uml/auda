from decimal import Decimal

from sqlalchemy import CheckConstraint, Numeric, Text
from sqlalchemy.orm import Mapped, mapped_column

from .base_model import SABase


# Canonical unit labels present in the frozen PRISM data_extractions export.
TARGET_UNITS = (
    'metric ton', 'people', '%', 'USD', 'kg/capita', 'people/sq. km'
)


class UnitConversion(SABase):
    """Stores operator-configured conversions into PRISM canonical units.

    A conversion multiplies the source value by ``factor``. Rules are direct,
    case-sensitive mappings, not chains. Canonical units need no table rows.
    Operators must verify that each source and target measure the same quantity.

    Attributes:
        source_unit: Exact source label, without surrounding whitespace.
        target_unit: Canonical unit label used by the analytical data.
        factor: Positive, finite multiplier applied before numeric coercion.
    """

    __tablename__ = 'unit_conversion'
    __table_args__ = (
        CheckConstraint(
            "source_unit = trim(source_unit) AND source_unit <> '' "
            "AND source_unit <> 'NA'",
            name='unit_conversion_source_label',
        ),
        CheckConstraint(
            "target_unit IN ('metric ton', 'people', '%', 'USD', "
            "'kg/capita', 'people/sq. km')",
            name='unit_conversion_target_label',
        ),
        CheckConstraint(
            "source_unit NOT IN ('metric ton', 'people', '%', 'USD', "
            "'kg/capita', 'people/sq. km')",
            name='unit_conversion_preserve_canonical',
        ),
        CheckConstraint(
            "factor > 0 AND factor < 'Infinity'::numeric",
            name='unit_conversion_positive_finite_factor',
        ),
    )

    source_unit: Mapped[str] = mapped_column(Text, primary_key=True)
    target_unit: Mapped[str] = mapped_column(Text, nullable=False)
    factor: Mapped[Decimal] = mapped_column(Numeric, nullable=False)
