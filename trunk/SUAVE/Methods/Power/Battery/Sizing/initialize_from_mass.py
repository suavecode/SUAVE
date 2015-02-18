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

def initialize_from_mass(battery, mass): #adds a battery that is optimized based on power and energy requirements and technology
    battery.mass_properties.mass=mass
    battery.max_energy          =mass*battery.specific_energy
    battery.max_power           =mass*battery.specific_power
    
    
    return