-- Copyright 2025 AUDA Contributors
--
-- This SQL file creates metadata for tables.
-- Read the documentation for more details.

--------------------------------------------------------------------------------

TRUNCATE TABLE table_metadata RESTART IDENTITY;

INSERT INTO table_metadata (name, type)
VALUES
('table_metadata', 'system'),
('data_column_metadata', 'system'),
('d_demography', 'data'),
('d_plastic_resin', 'data'),
('d_waste_generation_management', 'data');

