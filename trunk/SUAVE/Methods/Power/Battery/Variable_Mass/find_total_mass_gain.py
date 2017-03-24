## @ingroup methods-power-battery-variable_mass
# find_total_mass_gain.py
# 
# Created:  ### 2104, M. Vegh
# Modified: Feb 2016, E. Botero

# ----------------------------------------------------------------------
#  Find Total Mass Gain
# ----------------------------------------------------------------------
## @ingroup methods-power-battery-variable_mass
def find_total_mass_gain(battery):
    """finds the total mass of air that the battery 
    accumulates when discharged fully
    
    Assumptions:
    Earth Atmospheric composition
    
    Inputs:
    battery.max_energy [J]
    battery.
      mass_gain_factor [kg/W]
      
    Outputs:
      mdot             [kg]
    """
    
    
    
    
    mgain=battery.max_energy*battery.mass_gain_factor
    
    return mgain