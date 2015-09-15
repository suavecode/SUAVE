'''determines mass,  from a ragone curve correlation'''
#by M. Vegh
#Created 2014
#Modified September 2015
""" SUAVE Methods for Energy Systems """

# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------

from SUAVE.Methods.Power.Battery.Sizing import initialize_from_energy_and_power
from find_specific_power import find_specific_power
# ----------------------------------------------------------------------
#  Methods
# ----------------------------------------------------------------------

def find_ragone_properties(specific_energy, battery, energy, power): #modifies battery based on ragone plot characteristics of battery
    
    find_specific_power(battery, specific_energy)
    initialize_from_energy_and_power(battery, energy, power)
    
    return battery.mass_properties.mass #used for a simple optimization