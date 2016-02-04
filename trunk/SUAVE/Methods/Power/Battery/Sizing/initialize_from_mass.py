# initialize_from_mass.py
# 
# Created:  ### ####, M. Vegh
# Modified: Feb 2016, E. Botero

# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------

import autograd.numpy as np 

# ----------------------------------------------------------------------
#  Methods
# ----------------------------------------------------------------------

def initialize_from_mass(battery, mass):
    battery.mass_properties.mass = mass
    battery.max_energy           = mass*battery.specific_energy
    battery.max_power            = mass*battery.specific_power