#Created : M. Vegh 4/23/15
#Modified:M. Vegh, September 2015
""" Calculates mass flow of fuel cell based on method from Larminie and 
Dicks (Fuel Cell Systems Explained) """

# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------


import SUAVE
from SUAVE.Core import Units
from find_power_larminie import find_power_larminie
# ----------------------------------------------------------------------
#  Methods
# ----------------------------------------------------------------------

def find_power_diff_larminie(current_density, fuel_cell, power_desired): #adds a battery that is optimized based on power and energy requirements and technology 
                                    
    power_out     = find_power_larminie(current_density, fuel_cell)              #obtain power output in W
    
    #want to minimize
    return abs(power_desired-power_out)