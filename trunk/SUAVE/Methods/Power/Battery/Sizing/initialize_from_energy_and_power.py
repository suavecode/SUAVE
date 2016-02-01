# initialize_from_energy_and_power.py
# 
# Created:  Feb 2015, M. Vegh
# Modified: Feb 2016, M. Vegh
#           Feb 2016, E. Botero

# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------

import numpy as np

# ----------------------------------------------------------------------
#  Initialize from Energy and Power
# ----------------------------------------------------------------------

def initialize_from_energy_and_power(battery, energy, power, max='hard'):
    
    energy_mass = energy/battery.specific_energy
    power_mass  = power/battery.specific_power
    mass        = np.maximum(energy_mass, power_mass)
    
    if max=='soft': #use softmax function (makes it differentiable)
        #make it so the exponentials are taking ~10 (closer numerically, while still differentiable)
        scaling = 10.**(np.floor(np.log10(mass))-1) 
        
        #write it this way to prevent overflow
        mass=scaling*np.log(np.exp(energy_mass/scaling)+np.exp(power_mass/scaling))  

    battery.mass_properties.mass = mass
    battery.max_energy           = battery.specific_energy*mass
    battery.max_power            = battery.specific_power*mass