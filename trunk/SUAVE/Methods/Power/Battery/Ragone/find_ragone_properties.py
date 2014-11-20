'''determines mass,  from a ragone curve correlation'''
#by M. Vegh

""" SUAVE Methods for Energy Systems """

# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------

import numpy as np
from SUAVE.Methods.Power.Battery.Sizing import initialize_from_energy_and_power
import find_specific_power
# ----------------------------------------------------------------------
#  Methods
# ----------------------------------------------------------------------

def find_ragone_properties(battery, specific_energy,specific_power, energy, power): #adds a battery that is optimized based on power and energy requirements and technology
    find_specific_power(battery, specific_energy)
    initialize_from_energy_and_power(battery, energy, power)
    
    return