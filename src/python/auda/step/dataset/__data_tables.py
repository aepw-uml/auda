class DataTableName:
    """Names of data tables in the AUDA database."""

    DEMOGRAPHY = 'd_demography'
    WASTE_GENERATION_MANAGEMENT = 'd_waste_generation_management'
    PLASTIC_RESIN_DATA = 'd_plastic_resin_data'


class DemographyColumn:
    """Data columns in the "d_demography" data table."""

    YEAR = f'{DataTableName.DEMOGRAPHY}.year'
    LOCATION = f'{DataTableName.DEMOGRAPHY}.location'
    POPULATION = f'{DataTableName.DEMOGRAPHY}.population'
    GDP = f'{DataTableName.DEMOGRAPHY}.gdp'
    URBAN_POPULATION = f'{DataTableName.DEMOGRAPHY}.urban_population'
    RURAL_POPULATION = f'{DataTableName.DEMOGRAPHY}.rural_population'


class WasteGenerationManagementColumn:
    """Data columns in the "d_waste_generation_management" data table."""

    YEAR = f'{DataTableName.WASTE_GENERATION_MANAGEMENT}.year'
    LOCATION = f'{DataTableName.WASTE_GENERATION_MANAGEMENT}.location'
    PLASTIC_WASTE_GENERATED = (
        f'{DataTableName.WASTE_GENERATION_MANAGEMENT}.plastic_waste_generated'
    )
    PLASTIC_WASTE_GENERATED_PER_CAPITA = (
        f'{DataTableName.WASTE_GENERATION_MANAGEMENT}.'
        'plastic_waste_generated_per_capita'
    )
    PLASTIC_MISMANAGED = (
        f'{DataTableName.WASTE_GENERATION_MANAGEMENT}.plastic_mismanaged'
    )
    PLASTIC_MISMANAGED_PERCENT = f'{DataTableName.WASTE_GENERATION_MANAGEMENT}.plastic_mismanaged_percent'
    PLASTIC_WASTE_COLLECTED_PERCENT = f'{DataTableName.WASTE_GENERATION_MANAGEMENT}.plastic_waste_collected_percent'


class PlasticResinDataColumn:
    """Data columns in the "d_plastic_resin_data" data table."""

    YEAR = f'{DataTableName.PLASTIC_RESIN_DATA}.year'
    LOCATION = f'{DataTableName.PLASTIC_RESIN_DATA}.location'
