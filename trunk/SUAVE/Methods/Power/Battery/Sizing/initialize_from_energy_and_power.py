"""sizes battery based on total battery properties"""
#by M. Vegh

""" SUAVE Methods for Energy Systems """

# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------

import numpy as np
# ----------------------------------------------------------------------
#  Methods
# ----------------------------------------------------------------------

def initialize_from_energy_and_power(battery, energy, power): #adds a battery that is optimized based on power and energy requirements and technology
    
    
    mass=np.maximum(energy/battery.specific_energy, power/battery.specific_power)
    battery.mass_properties.mass=mass
    battery.max_energy=battery.specific_energy*mass
    battery.max_power =battery.specific_power*mass
    
    return