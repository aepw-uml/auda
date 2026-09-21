"""Checks configurable unit conversion without connecting to a database.

Run with PYTHONPATH=python and the usual AUDA environment variables configured:
python -m unittest discover -s python/tests -p 'test_unit_conversion.py'
"""

import unittest
from decimal import Decimal
from logging import getLogger
from unittest.mock import Mock

from sqlalchemy import Column, Integer, MetaData, Table
from util.database.model import (
    DataColumnMetadata,
    PrismDataPoint,
    PrismLocation,
)
from util.database.model.unit_conversion import TARGET_UNITS, UnitConversion
from util.migration.datapoint import Datapoint
from util.migration.migration_service import MigrationService
from util.migration.unit_converter import UnitConverter


class UnitConversionTest(unittest.TestCase):
    """Checks unit conversion and its position in the migration pipeline."""

    def test_empty_rules_preserve_canonical_values(self) -> None:
        """Preserves every canonical value, including its numeric formatting."""

        converter = UnitConverter([])
        for unit in TARGET_UNITS:
            with self.subTest(unit=unit):
                self.assertEqual(
                    converter.convert('1,234.50', f' {unit} '),
                    ('1,234.50', unit),
                )

    def test_conversion_and_alias(self) -> None:
        """Converts numeric representations and explicit unit aliases."""

        converter = UnitConverter([
            UnitConversion(
                source_unit='kg', target_unit='metric ton',
                factor=Decimal('0.001'),
            ),
            UnitConversion(
                source_unit='persons', target_unit='people', factor=Decimal(1),
            ),
        ])
        for value in ('1,250', '1.25E3'):
            converted, unit = converter.convert(value, 'kg')
            self.assertEqual(Decimal(converted), Decimal('1.25'))
            self.assertEqual(unit, 'metric ton')
        self.assertEqual(converter.convert('12', 'persons'), ('12', 'people'))
        for value in ('NaN', 'Infinity', 'invalid'):
            with self.subTest(value=value), self.assertRaises(ValueError):
                converter.convert(value, 'kg')

    def test_unknown_units_fail(self) -> None:
        """Rejects unmapped and unavailable units without guessing aliases."""

        for unit in ('kg', 'Metric Ton', '', 'NA'):
            with self.subTest(unit=unit), self.assertRaises(ValueError):
                UnitConverter([]).convert('100', unit)

    def test_invalid_rules_fail(self) -> None:
        """Rejects invalid factors, target labels, and canonical overrides."""

        for source, target, factor in (
            ('kg', 'metric ton', '0'),
            ('kg', 'metric ton', '-1'),
            ('kg', 'metric ton', 'NaN'),
            ('kg', 'metric ton', 'Infinity'),
            ('kg', 'unknown', '1'),
            ('USD', 'metric ton', '1'),
            (' kg ', 'metric ton', '1'),
            ('NA', 'metric ton', '1'),
        ):
            with self.subTest(source=source, target=target, factor=factor):
                with self.assertRaises(ValueError):
                    UnitConverter([UnitConversion(
                        source_unit=source, target_unit=target,
                        factor=Decimal(factor),
                    )])

    def test_migration_converts_before_integer_coercion(self) -> None:
        """Converts fractional source values before integer coercion."""

        service = MigrationService.__new__(MigrationService)
        service.logger = getLogger('unit-conversion-test')
        service.unit_converter = UnitConverter([UnitConversion(
            source_unit='thousand people', target_unit='people',
            factor=Decimal(1000),
        )])
        service.insert_or_update_datapoint = Mock(return_value=True)
        table = Table('population', MetaData(), Column('value', Integer))
        arguments = (
            {'indicator': PrismDataPoint(name='Population')},
            {'location': PrismLocation(location_name='Example')},
            {'Population': DataColumnMetadata(
                table_name='population', column_name='value',
            )},
            {'population': table},
        )
        datapoint = Datapoint(
            'record', 'indicator', 'location', 2020, '1.25', 'thousand people'
        )
        self.assertTrue(service.migrate_datapoint(datapoint, *arguments))
        converted = service.insert_or_update_datapoint.call_args.args[-1]
        self.assertEqual(service._coerce_for_column(Integer(), converted), 1250)
        service.insert_or_update_datapoint.reset_mock()
        unmapped = Datapoint(
            'record', 'indicator', 'location', 2020, '1.25', 'unknown'
        )
        with self.assertRaisesRegex(ValueError, 'Cannot migrate extraction'):
            service.migrate_datapoint(unmapped, *arguments)
        service.insert_or_update_datapoint.assert_not_called()


if __name__ == '__main__':
    unittest.main()
