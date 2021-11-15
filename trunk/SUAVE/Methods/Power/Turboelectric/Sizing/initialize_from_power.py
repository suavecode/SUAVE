## @ingroup Methods-Power-Turboelectric-Sizing

# initialize_from_power.py
#
# Created : Nov 2019, K. Hamilton
# Modified: Nov 2021,   S. Claridge

# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------
from SUAVE.Core import Units

# ----------------------------------------------------------------------
#  Initialize from Power
# ----------------------------------------------------------------------

## @ingroup Methods-Power-Turboelectric-Sizing
def initialize_from_power(turboelectric,power,conditions):
    '''
    assigns the mass of a single turboelectric generator based on the power and specific power
    Assumptions:
    Power output is derated relative to the air pressure. 100% power is considered a mean sea level.
    
    Inputs:
    power                 [J]
    turboelectric.
      specific_power      [W/kg]
    conditions.
      freestream_pressure [Pa]
    
    
    Outputs:
    turboelectric.
      mass_properties.
        mass         [kg]
    '''

    # Unpack inputs
    pressure            = conditions.freestream.pressure
    specific_power      = turboelectric.specific_power

    # Ambient pressure as proportion of sealevel pressure, for use in derating the gas turbine
    derate              = pressure/101325. 
    
    # Proportionally increase demand relative to the sizing altitude
    demand_power        = power/derate

    # Apply specific power specification to size individual powersupply
    powersupply_mass    = demand_power/specific_power

    # Store sized turboelectric rated power
    turboelectric.rated_power   = power

    # Modify turboelectric to include the newly created mass data
    turboelectric.mass_properties.mass  = powersupply_mass
