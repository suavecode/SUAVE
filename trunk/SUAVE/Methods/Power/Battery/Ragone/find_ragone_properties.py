'''determines mass,  from a ragone curve correlation'''
#by M. Vegh

""" SUAVE Methods for Energy Systems """

# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------

import numpy as np
from SUAVE.Methods.Power.Battery.Sizing import initialize_from_energy_and_power
from find_specific_power import find_specific_power
# ----------------------------------------------------------------------
#  Methods
# ----------------------------------------------------------------------

def find_ragone_properties(battery, specific_energy, energy, power): #modifies battery based on ragone plot characteristics of battery
    \
    find_specific_power(battery, specific_energy)
    initialize_from_energy_and_power(battery, energy, power)
    
    return battery.mass_properties.mass