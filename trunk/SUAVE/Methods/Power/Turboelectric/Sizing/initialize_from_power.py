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
def initialize_from_power(turboelectric,power):
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
    turboelectric.mass_properties.mass = power/turboelectric.specific_power
