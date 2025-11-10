TRUNCATE TABLE data_column_metadata RESTART IDENTITY;

INSERT INTO data_column_metadata (
  table_name, column_name, original_column_name, data_type
)
VALUES
-- Table `d_demography`
('d_demography', 'population', 'Population', 'integer'),
('d_demography', 'gdp', 'Gross Domestic Product (GDP)', 'integer'),
('d_demography', 'population_density', 'Population Density', 'integer'),
('d_demography', 'tourist_population', 'Tourist Population', 'integer'),
('d_demography', 'rural_population', 'Rural Population', 'integer'),
('d_demography', 'urban_population', 'Urban Population', 'integer'),
('d_demography', 'coastal_population', 'Coastal Population', 'integer'),
-- Table `d_waste_generation_management`
(
  'd_waste_generation_management',
  'plastic_waste_generated',
  'Plastic Waste Generated',
  'float'
),
(
  'd_waste_generation_management',
  'plastic_waste_generated_per_capita',
  'Plastic Waste Generated Per Capita',
  'float'
),
(
  'd_waste_generation_management',
  'plastic_mismanaged',
  'Plastic Mismanaged',
  'float'
),
(
  'd_waste_generation_management',
  'plastic_mismanaged_percent',
  'Plastic Mismanaged %',
  'float'
),
(
  'd_waste_generation_management',
  'plastic_waste_collected',
  'Plastic Waste Collected',
  'float'
),
(
  'd_waste_generation_management',
  'plastic_waste_collected_percent',
  'Plastic Waste % Collected',
  'float'
),
(
  'd_waste_generation_management',
  'plastic_waste_collected_for_recycling_percent',
  'Plastic Waste % Collected For Recycling',
  'float'
),
(
  'd_waste_generation_management',
  'plastic_waste_formally_collected',
  'Plastic Waste Formally Collected',
  'float'
),
(
  'd_waste_generation_management',
  'plastic_waste_formally_collected_percent',
  'Plastic Waste % Formally Collected',
  'float'
),
(
  'd_waste_generation_management',
  'plastic_waste_informally_collected',
  'Plastic Waste Informally Collected',
  'float'
),
(
  'd_waste_generation_management',
  'plastic_waste_informally_collected_percent',
  'Plastic Waste % Informally Collected',
  'float'
),
(
  'd_waste_generation_management',
  'incineration_percent',
  'Incineration',
  'float'
),
(
  'd_waste_generation_management',
  'landfill_percent',
  'Landfill',
  'float'
),
(
  'd_waste_generation_management',
  'plastic_leakage',
  'Plastic Leakage',
  'float'
),
(
  'd_waste_generation_management',
  'plastic_leakage_percent',
  'Plastic Leakage %',
  'float'
),
-- Table `d_plastic_resin`
(
  'd_plastic_resin',
  'plastic_percent',
  'Plastic %',
  'float'
),
(
  'd_plastic_resin',
  'total_plastic_consumption',
  'Total Plastic Consumption',
  'float'
),
(
  'd_plastic_resin',
  'total_resin_consumption',
  'Total Resin Consumption',
  'float'
),
(
  'd_plastic_resin',
  'polymer_pp_consumption',
  'Polymer PP Consumed',
  'float'
),
(
  'd_plastic_resin',
  'polymer_hdpe_consumption',
  'Polymer HDPE Consumed',
  'float'
),
(
  'd_plastic_resin',
  'polymer_ldpe_consumption',
  'Polymer LDPE Consumed',
  'float'
),
(
  'd_plastic_resin',
  'polymer_ps_consumption',
  'Polymer PS Consumed',
  'float'
),
(
  'd_plastic_resin',
  'polymer_pet_consumption',
  'Polymer PET Consumed',
  'float'
),
(
  'd_plastic_resin',
  'polymer_pvc_consumption',
  'Polymer PVC Consumed',
  'float'
),
(
  'd_plastic_resin',
  'polymer_other_consumption',
  'Other Polymer Consumed',
  'float'
),
(
  'd_plastic_resin',
  'resin_pp_consumption',
  'Resin PP Consumed',
  'float'
),
(
  'd_plastic_resin',
  'resin_hdpe_consumption',
  'Resin HDPE Consumed',
  'float'
),
(
  'd_plastic_resin',
  'resin_ldpe_consumption',
  'Resin LDPE Consumed',
  'float'
),
(
  'd_plastic_resin',
  'resin_ps_consumption',
  'Resin PS Consumed',
  'float'
),
(
  'd_plastic_resin',
  'resin_pet_consumption',
  'Resin PET Consumed',
  'float'
),
(
  'd_plastic_resin',
  'resin_pvc_consumption',
  'Resin PVC Consumed',
  'float'
),
(
  'd_plastic_resin',
  'resin_other_consumption',
  'Other Resin Consumed',
  'float'
);

