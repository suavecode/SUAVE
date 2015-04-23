
#by M. Vegh

""" SUAVE Methods for Energy Systems """

# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------

import numpy as np

# ----------------------------------------------------------------------
#  Methods
# ----------------------------------------------------------------------

def zero_fidelity(fuel_cell,conditions,numerics): #adds a battery that is optimized based on power and energy requirements and technology
    power  = fuel_cell.inputs.power_in
    mdot        = power/(self.propellant.specific_energy*self.efficiency)                      #mass flow rate of the fuel  
  
    return [mdot]