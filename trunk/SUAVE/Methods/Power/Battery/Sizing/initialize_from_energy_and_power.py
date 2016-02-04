# initialize_from_energy_and_power.py
# 
# Created:  Feb 2015, M. Vegh
# Modified: Feb 2016, M. Vegh
#           Feb 2016, E. Botero
#           Feb 2016, M. Vegh

# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------

import autograd.numpy as np 
from SUAVE.Methods.Utilities.soft_max import soft_max
# ----------------------------------------------------------------------
#  Initialize from Energy and Power
# ----------------------------------------------------------------------

def initialize_from_energy_and_power(battery, energy, power, max='hard'):
    
    energy_mass = energy/battery.specific_energy
    power_mass  = power/battery.specific_power
    
    if max=='soft': #use softmax function (makes it differentiable)
        mass=soft_max(energy_mass,power_mass)
        
    else:
        mass=np.maximum(energy_mass, power_mass)

    battery.mass_properties.mass = mass
    battery.max_energy           = battery.specific_energy*mass
    battery.max_power            = battery.specific_power*mass