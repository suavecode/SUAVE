# find_ragone_properties.py
# 
# Created:  ### 2104, M. Vegh
# Modified: Sep 2105, M. Vegh
#           Feb 2016, E. Botero

# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------

from SUAVE.Methods.Power.Battery.Sizing import initialize_from_energy_and_power
from find_specific_power import find_specific_power

# ----------------------------------------------------------------------
#  Find Ragone Properties
# ----------------------------------------------------------------------

def find_ragone_properties(specific_energy, battery, energy, power):
    '''determines mass,  from a ragone curve correlation'''
    
    find_specific_power(battery, specific_energy)
    initialize_from_energy_and_power(battery, energy, power)
    
    #used for a simple optimization
    return battery.mass_properties.mass 