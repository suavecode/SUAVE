"""sizes an optimized battery based on power and energy requirements based on a Ragone plot curve fit"""
#by M. Vegh
#Modified 1-27-2015
""" SUAVE Methods for Energy Systems """

# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------

import numpy as np
import scipy as sp
from find_ragone_properties import find_ragone_properties
# ----------------------------------------------------------------------
#  Methods
# ----------------------------------------------------------------------

def find_ragone_optimum(battery, energy, power): #adds a battery that is optimized based on power and energy requirements and technology
    """Inputs:
            battery
            energy= energy battery is required to hold [J]
            power= power battery is required to provide [W]

       Reads:
            battery.type
            battery.specific_energy
            battery.specific_power
            battery.ragone.constant_1
            battery.ragone.constant_2
            battery.ragone.upper_bound
            battery.ragone.lower_bound
            energy
            power

       Outputs:
            battery.specific_energy
            battery.specific_power
            battery.mass_properties.mass
    """

    specific_energy_guess=battery.specific_energy
    lb=battery.ragone.lower_bound
    ub=battery.ragone.upper_bound

    #optimize!
    specific_energy_opt=sp.optimize.fminbound(find_ragone_properties, lb, ub, args=( battery, energy, power))
    
    #now initialize the battery with the new optimum properties
    find_ragone_properties(specific_energy_opt, battery, energy, power)
    return