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
    battery.max_energy=energy
    battery.max_power =power
    
    battery.mass=np.max(energy/battery.specific_energy, power/power.specific_power)
    
    
    return