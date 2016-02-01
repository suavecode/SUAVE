# find_power_larminie.py
#
# Created : Apr 2015, M. Vegh 
# Modified: Sep 2015, M. Vegh
#           Feb 2016, E. Botero

# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------

import numpy as np
from SUAVE.Core import Units
from find_voltage_larminie import find_voltage_larminie

# ----------------------------------------------------------------------
#  Find Power Larminie
# ----------------------------------------------------------------------

def find_power_larminie(current_density, fuel_cell, sign=1.0):
    
    # sign variable is used so that you can maximize the power, by minimizing the -power
    i1            = current_density/(Units.mA/(Units.cm**2.)) 
    A             = fuel_cell.interface_area/(Units.cm**2.)
    v             = find_voltage_larminie(fuel_cell,current_density)  #useful voltage vector
    power_out     = sign* np.divide(np.multiply(v,i1),1000.0)*A       #obtain power output in W/cell
    
    #want to minimize
    return power_out