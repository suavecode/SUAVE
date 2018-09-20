## @ingroup Methods-Power-Fuel_Cell-Discharge
# find_power_diff_larminie.py
#
# Created : Apr 2015, M. Vegh 
# Modified: Sep 2015, M. Vegh
#           Feb 2016, E. Botero

# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------

from .find_power_larminie import find_power_larminie

# ----------------------------------------------------------------------
#  Find Power Difference Larminie
# ----------------------------------------------------------------------
## @ingroup Methods-Power-Fuel_Cell-Discharge
def find_power_diff_larminie(current_density, fuel_cell, power_desired):
    '''
    function that determines the power difference between the actual power
    and a desired input power, based on an input current density

    Assumptions:
    None
    
    Inputs:
    current_density                [Amps/m**2]
    power_desired                  [Watts]
    fuel_cell
      
    
    Outputs
    (power_desired-power_out)**2   [Watts**2]
    '''
    #obtain power output in W
    
    power_out     = find_power_larminie(current_density, fuel_cell)              
    
    #want to minimize
    return (power_desired-power_out)**2.#abs(power_desired-power_out)