## @ingroup methods-power-battery-sizing

## @ingroup Methods-Power-Battery-Sizing
# initialize_from_energy_and_power.py
# 
# Created:  Feb 2015, M. Vegh
# Modified: Feb 2016, M. Vegh
#           Feb 2016, E. Botero
#           Feb 2016, M. Vegh

# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------

import numpy as np
from SUAVE.Methods.Utilities.soft_max import soft_max
# ----------------------------------------------------------------------
#  Initialize from Energy and Power
# ----------------------------------------------------------------------
## @ingroup Methods-Power-Battery-Sizing
def initialize_from_energy_and_power(battery, energy, power, max='hard'):
    """
    Uses a soft_max function to calculate the batter mass, maximum energy, and maximum power
    from the energy and power requirements, as well as the specific energy and specific power
    
    Assumptions:
    None
    
    Inputs:
    energy            [J]
    power             [W]
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
    
    energy_mass = energy/battery.specific_energy
    power_mass  = power/battery.specific_power
    
    if max=='soft': #use softmax function (makes it differentiable)
        mass=soft_max(energy_mass,power_mass)
        
    else:
        mass=np.maximum(energy_mass, power_mass)

    battery.mass_properties.mass = mass
    battery.max_energy           = battery.specific_energy*mass
    battery.max_power            = battery.specific_power*mass