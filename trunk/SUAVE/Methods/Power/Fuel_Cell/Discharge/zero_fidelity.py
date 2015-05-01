#Created by M. Vegh 4/23/15

""" Calculates mass flow of fuel cell based solely on specific energy
end base efficiency factor """

# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------

import numpy as np

# ----------------------------------------------------------------------
#  Methods
# ----------------------------------------------------------------------

def zero_fidelity(fuel_cell,conditions,numerics): #adds a battery that is optimized based on power and energy requirements and technology
    power       = fuel_cell.inputs.power_in
    mdot        = power/(fuel_cell.propellant.specific_energy*fuel_cell.efficiency)                      #mass flow rate of the fuel  
  
    return mdot