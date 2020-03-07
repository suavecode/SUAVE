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
def initialize_from_power(turboelectric,number_of_powersupplies,power,conditions):
    '''
    assigns the mass of the turboelectric generator based on the power and specific power
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
    
    # Divide power evenly between available powersupplies.
    individual_demand   = power/number_of_powersupplies
    # Proportionally increase demand relative to the sizing altitude
    demand_power        = individual_demand/derate

    # Apply specific power specification to size individual powersupply
    powersupply_mass    = demand_power/specific_power
    
    # Multiply mass by number of powersupplies to give the total mass of all powersupplies
    mass                = number_of_powersupplies*powersupply_mass

    # Modify turboelectric to include the newly created mass data
    turboelectric.mass_properties.mass  = mass