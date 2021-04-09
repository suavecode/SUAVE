## @ingroup Methods-Power-Fuel_Cell-Discharge
# larminie.py
#
# Created : Apr 2015, M. Vegh 
# Modified: Feb 2016, E. Botero
  
# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------

import numpy as np
import scipy as sp
from SUAVE.Core import Units
from .find_voltage_larminie import find_voltage_larminie
from .find_power_diff_larminie import find_power_diff_larminie

# ----------------------------------------------------------------------
#  Larminie
# ----------------------------------------------------------------------

## @ingroup Methods-Power-Fuel_Cell-Discharge
def larminie(fuel_cell,conditions,numerics):
    '''
    function that determines the mass flow rate based on a required power input
    
    Assumptions:
    None (calls other functions)
    
    Inputs:
    fuel_cell.
        inputs.
            power_in       [W]
        ideal_voltage      [V] 
    
    Outputs:
        mdot               [kg/s]
     
    
    
    
    '''
    
    
    power           = fuel_cell.inputs.power_in  
    lb              = .1*Units.mA/(Units.cm**2.)    #lower bound on fuel cell current density
    ub              = 1200.0*Units.mA/(Units.cm**2.)
    current_density = np.zeros_like(power)
    
    for i in range(len(power)):
        current_density[i] = sp.optimize.fminbound(find_power_diff_larminie, lb, ub, args=(fuel_cell, power[i]))
    
    v          = find_voltage_larminie(fuel_cell,current_density)    
    efficiency = np.divide(v, fuel_cell.ideal_voltage)
    mdot       = np.divide(power,np.multiply(fuel_cell.propellant.specific_energy,efficiency))

    print('efficiency=', efficiency)
    print('current_density=', current_density)
    print('v=', v)
   
    return mdot