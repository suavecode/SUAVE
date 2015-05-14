#Created by M. Vegh 4/23/15

""" Calculates mass flow of fuel cell based on method from Larminie and 
Dicks (Fuel Cell Systems Explained) """

# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------

import numpy as np
import SUAVE
from SUAVE.Core import Units
from find_voltage_larminie import find_voltage_larminie
# ----------------------------------------------------------------------
#  Methods
# ----------------------------------------------------------------------

def find_power_larminie(current_density, fuel_cell, sign=1.0): #adds a battery that is optimized based on power and energy requirements and technology
    #sign variable is used so that you can maximize the power, by minimizing the -power
    i1            = current_density/(Units.mA/(Units.cm**2.)) 
    A             = fuel_cell.interface_area/(Units.cm**2.)
    v             = find_voltage_larminie(fuel_cell,current_density)                                          #useful voltage vector
    power_out     = sign* np.divide(np.multiply(v,i1),1000.0)*A              #obtain power output in W/cell
    
    #want to minimize
    return power_out