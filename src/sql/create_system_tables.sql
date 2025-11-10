-- Copyright 2025 ADA Contributors
--
-- This scripting file creates some system tables for the ADA application.
-- Read the documentation for more details.

--------------------------------------------------------------------------------

CREATE TABLE table_metadata (
  id INTEGER GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
  name VARCHAR(255) NOT NULL UNIQUE,
  type VARCHAR(16) NOT NULL CHECK (type IN ('system', 'data'))
);

COMMENT ON TABLE table_metadata
IS 'This table stores metadata about all tables in the system that can be '
'accessed by the client.';

COMMENT ON COLUMN table_metadata.name
IS 'The name of the table, which is used to reference it in queries.';

COMMENT ON COLUMN table_metadata.type
IS 'the type of the table, either "system" for system tables or "data" for'
'data tables.';

--------------------------------------------------------------------------------

CREATE TABLE data_column_metadata (
  id INTEGER GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
  table_name VARCHAR(255) NOT NULL,
  column_name VARCHAR(255) NOT NULL,
  original_column_name VARCHAR(255) NOT NULL,
  data_type VARCHAR(64) NOT NULL,
  unit VARCHAR(64) NOT NULL DEFAULT '',
  description TEXT NOT NULL DEFAULT '',

  UNIQUE (table_name, column_name)
);

COMMENT ON TABLE data_column_metadata
IS 'This table stores metadata of columns in data tables.';

COMMENT ON COLUMN data_column_metadata.table_name
IS 'The name of the data table this column belongs to.';

COMMENT ON COLUMN data_column_metadata.column_name
IS 'The name of the column in the data table.';

COMMENT ON COLUMN data_column_metadata.original_column_name
IS 'The column name in the original database.';

COMMENT ON COLUMN data_column_metadata.data_type
IS 'The data type of this column, such as "float" and "string"';

COMMENT ON COLUMN data_column_metadata.unit
IS 'The unit of measurement for this column.';

COMMENT ON COLUMN data_column_metadata.description
IS 'The description associated with this column from the original database.';

--------------------------------------------------------------------------------

CREATE TABLE unit_conversion_coefficients (
  id INTEGER GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
  source_unit VARCHAR(64) NOT NULL,
  target_unit VARCHAR(64) NOT NULL,
  coefficient FLOAT NOT NULL
);

COMMENT ON TABLE unit_conversion_coefficients
IS 'This table stores unit conversion coefficients for different units. '
'It is used during migration to convert units from the original database.';

COMMENT ON COLUMN unit_conversion_coefficients.source_unit
IS 'The source unit for the conversion.';

COMMENT ON COLUMN unit_conversion_coefficients.target_unit
IS 'The target unit for the conversion.';

COMMENT ON COLUMN unit_conversion_coefficients.coefficient
IS 'The coefficient used to convert from the source unit to the target unit.';

