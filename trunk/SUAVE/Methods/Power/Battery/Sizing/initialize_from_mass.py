## @ingroup Methods-Power-Battery-Sizing
# initialize_from_mass.py
# 
# Created:  ### ####, M. Vegh
# Modified: Feb 2016, E. Botero

# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------

import numpy as np

# ----------------------------------------------------------------------
#  Methods
# ----------------------------------------------------------------------
## @ingroup Methods-Power-Battery-Sizing
def initialize_from_mass(battery, mass):
    """
    Calculate the max energy and power based of the mass
    Assumptions:
    A constant value of specific energy and power

    Inputs:
    mass              [kilograms]
    battery.
      specific_energy [J/kg]               
      specific_power  [W/kg]

    Outputs:
     battery.
       max_energy
       max_power
       mass_properties.
        mass


    """    
    battery.mass_properties.mass = mass
    battery.max_energy           = mass*battery.specific_energy
    battery.max_power            = mass*battery.specific_power