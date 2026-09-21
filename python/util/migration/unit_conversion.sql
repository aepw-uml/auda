-- Run in the AUDA analytical database before migration.
-- Leave this table empty for the existing frozen data. Populate source labels
-- and multipliers only after checking their meaning in the source records.
BEGIN;

CREATE TABLE IF NOT EXISTS unit_conversion (
    source_unit TEXT PRIMARY KEY,
    target_unit TEXT NOT NULL,
    factor NUMERIC NOT NULL,
    CONSTRAINT unit_conversion_source_label CHECK (
        source_unit = trim(source_unit) AND source_unit <> ''
        AND source_unit <> 'NA'
    ),
    CONSTRAINT unit_conversion_target_label CHECK (
        target_unit IN ('metric ton', 'people', '%', 'USD',
                        'kg/capita', 'people/sq. km')
    ),
    CONSTRAINT unit_conversion_preserve_canonical CHECK (
        source_unit NOT IN ('metric ton', 'people', '%', 'USD',
                           'kg/capita', 'people/sq. km')
    ),
    CONSTRAINT unit_conversion_positive_finite_factor CHECK (
        factor > 0 AND factor < 'Infinity'::numeric
    )
);

-- Register this as a system table, never an analytical data table.
LOCK TABLE table_metadata IN SHARE ROW EXCLUSIVE MODE;
INSERT INTO table_metadata (id, name, type)
SELECT COALESCE(MAX(id), 0) + 1, 'unit_conversion', 'system'
FROM table_metadata
HAVING NOT EXISTS (
    SELECT 1 FROM table_metadata WHERE name = 'unit_conversion'
);

COMMIT;

-- Example only: confirm the source label denotes kilograms of total mass.
-- INSERT INTO unit_conversion (source_unit, target_unit, factor)
-- VALUES ('kg', 'metric ton', 0.001);
-- Alias labels can use factor 1. Rules do not perform currency conversion,
-- infer dimensions, or resolve ambiguous labels such as "ton" automatically.
