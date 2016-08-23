# find_power_diff_larminie.py
#
# Created : Apr 2015, M. Vegh 
# Modified: Sep 2015, M. Vegh
#           Feb 2016, E. Botero

# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------

from find_power_larminie import find_power_larminie

# ----------------------------------------------------------------------
#  Find Power Difference Larminie
# ----------------------------------------------------------------------

def find_power_diff_larminie(current_density, fuel_cell, power_desired):
                                    
    #obtain power output in W
    power_out     = find_power_larminie(current_density, fuel_cell)              
    
    #want to minimize
    return abs(power_desired-power_out)