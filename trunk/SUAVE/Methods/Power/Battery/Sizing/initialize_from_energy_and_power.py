# initialize_from_energy_and_power.py
# 
# Created:  ### 2104, M. Vegh
# Modified: Jan 2016, E. Botero

# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------

import numpy as np

# ----------------------------------------------------------------------
#  Initialize from Energy and Power
# ----------------------------------------------------------------------

def initialize_from_energy_and_power(battery, energy, power):
    """sizes battery based on total battery properties"""
    
    mass = np.maximum(energy/battery.specific_energy, power/battery.specific_power)
    
    battery.mass_properties.mass = mass
    battery.max_energy           = battery.specific_energy*mass
    battery.max_power            = battery.specific_power*mass