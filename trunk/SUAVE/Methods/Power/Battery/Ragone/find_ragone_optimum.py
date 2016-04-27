# find_ragone_optimimum.py
# 
# Created:  ### 2104, M. Vegh
# Modified: Sep 2105, M. Vegh
#           Feb 2016, E. Botero

# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------

import scipy as sp
from find_ragone_properties import find_ragone_properties

# ----------------------------------------------------------------------
#  Methods
# ----------------------------------------------------------------------

def find_ragone_optimum(battery, energy, power): #adds a battery that is optimized based on power and energy requirements and technology
    """
    Uses Brent's Bracketing Method to find an optimum-mass battery based on the specific energy and specific power of the battery determined
    from the battery's ragone plot.
    
    Inputs:
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

    specific_energy_guess = battery.specific_energy
    
    lb = battery.ragone.lower_bound
    ub = battery.ragone.upper_bound

    #optimize!
    specific_energy_opt = sp.optimize.fminbound(find_ragone_properties, lb, ub, args=( battery, energy, power), xtol=1e-12)
    
    #now initialize the battery with the new optimum properties
    find_ragone_properties(specific_energy_opt, battery, energy, power)