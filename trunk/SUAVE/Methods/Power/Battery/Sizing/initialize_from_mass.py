# initialize_from_mass.py
# 
# Created:  ### ####, M. Vegh
# Modified: Jan 2016, E. Botero

# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------

import numpy as np

# ----------------------------------------------------------------------
#  Methods
# ----------------------------------------------------------------------

def initialize_from_mass(battery, mass):
    battery.mass_properties.mass = mass
    battery.max_energy           = mass*battery.specific_energy
    battery.max_power            = mass*battery.specific_power