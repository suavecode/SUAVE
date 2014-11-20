"""models discharge losses based on an empirical correlation"""
#by M. Vegh

""" SUAVE Methods for Energy Systems """

# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------

import numpy as np

# ----------------------------------------------------------------------
#  Methods
# ----------------------------------------------------------------------

def initialize_battery_from_mass(battery): #adds a battery that is optimized based on power and energy requirements and technology
    battery.max_energy=battery.mass_properties.mass*battery.specific_energy
    battery.max_power =battery.mass_properties.mass*battery.specific_power
    
    
    return