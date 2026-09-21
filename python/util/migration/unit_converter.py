from decimal import Decimal, InvalidOperation

from util.database.model.unit_conversion import TARGET_UNITS, UnitConversion


class UnitConverter:
    """Converts noncanonical source units using explicit database rules.

    Canonical values pass through unchanged even when the rule table is empty.
    Missing rules fail explicitly; ambiguous labels are never guessed.
    """

    def __init__(self, rules: list[UnitConversion]) -> None:
        """Validates and caches conversion rules for one migration run.

        Args:
            rules: Operator-configured conversion rows from the AUDA database.

        Raises:
            ValueError: If a rule is ambiguous or invalid.
        """

        self.rules: dict[str, tuple[str, Decimal]] = {}
        for rule in rules:
            source = rule.source_unit
            target = rule.target_unit
            factor = Decimal(str(rule.factor))
            if (
                not source
                or source != source.strip()
                or source == 'NA'
                or source in TARGET_UNITS
                or source in self.rules
                or target not in TARGET_UNITS
                or not factor.is_finite()
                or factor <= 0
            ):
                raise ValueError(
                    f'Invalid unit conversion rule for {source!r}.'
                )

            self.rules[source] = (target, factor)

    def convert(self, value: str, source_unit: str) -> tuple[str, str]:
        """Converts a value and returns its canonical unit.

        Args:
            value: Numeric source value, possibly with commas or an exponent.
            source_unit: Source label from the extraction record.

        Returns:
            The converted numeric string and canonical target unit.

        Raises:
            ValueError: If the unit is unmapped or a converted value is invalid.
        """

        source = source_unit.strip()
        if source in TARGET_UNITS:
            return value, source

        if source not in self.rules:
            raise ValueError(
                f'No unit conversion configured for {source!r}. Add a rule '
                'to the AUDA unit_conversion table before migration.'
            )

        target, factor = self.rules[source]
        try:
            numeric = Decimal(value.strip().replace(',', ''))
        except InvalidOperation as error:
            raise ValueError(f'Invalid numeric value {value!r}.') from error

        if not numeric.is_finite():
            raise ValueError(f'Non-finite numeric value {value!r}.')

        return str(numeric * factor), target
