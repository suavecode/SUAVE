## @ingroup Methods-Power-Turboelectric-Sizing

# initialize_from_power.py
#
# Created : Nov 2019, K. Hamilton

# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------
from SUAVE.Core import Units

# ----------------------------------------------------------------------
#  Initialize from Power
# ----------------------------------------------------------------------

## @ingroup Methods-Power-Turboelectric-Sizing
def initialize_from_power(turboelectric,number_of_powersupplies,power):
    '''
    assigns the mass of the turboelectric generator based on the power and specific power
    Assumptions:
    None
    
    Inputs:
    power            [J]
    turboelectric.
      specific_power [W/kg]
    
    
    Outputs:
    turboelectric.
      mass_properties.
        mass         [kg]
    '''
    individual_demand = power/number_of_powersupplies
    powersupply_mass = individual_demand/turboelectric.specific_power
    turboelectric.mass_properties.mass = number_of_powersupplies*powersupply_mass