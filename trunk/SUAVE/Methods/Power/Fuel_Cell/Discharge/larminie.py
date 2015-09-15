#Created by M. Vegh 4/23/15

""" Calculates mass flow of fuel cell based on method from Larminie and 
Dicks (Fuel Cell Systems Explained) """

# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------

import numpy as np
import scipy as sp
import SUAVE
from SUAVE.Core import Units
from find_power_larminie   import find_power_larminie
from find_voltage_larminie import find_voltage_larminie
from find_power_diff_larminie import find_power_diff_larminie
# ----------------------------------------------------------------------
#  Methods
# ----------------------------------------------------------------------

def larminie(fuel_cell,conditions,numerics): #adds a battery that is optimized based on power and energy requirements and technology
    power           = fuel_cell.inputs.power_in  
    lb              =.1*Units.mA/(Units.cm**2.)    #lower bound on fuel cell current density
    ub              =1200.0*Units.mA/(Units.cm**2.)
    current_density =np.zeros_like(power)
    for i in range(len(power)):
        current_density[i] =sp.optimize.fminbound(find_power_diff_larminie, lb, ub, args=(fuel_cell, power[i]))
    v               =find_voltage_larminie(fuel_cell,current_density)
    efficiency      =np.divide(v, fuel_cell.ideal_voltage)
    print 'efficiency=', efficiency
    print 'current_density=', current_density
    print 'v=', v
    mdot            = np.divide(power,np.multiply(fuel_cell.propellant.specific_energy,efficiency))
    
   
    return mdot