# initialize_from_energy_and_power

# Created:  Feb 2015, M. Vegh
# Modified: Feb 2016, M. Vegh


""" SUAVE Methods for Energy Systems """

# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------

import numpy as np
# ----------------------------------------------------------------------
#  Methods
# ----------------------------------------------------------------------

def initialize_from_energy_and_power(battery, energy, power, max='hard'): #adds a battery that is optimized based on power and energy requirements and technology
    energy_mass=energy/battery.specific_energy
    power_mass=power/battery.specific_power
    mass=np.maximum(energy_mass, power_mass)
    if max=='soft': #use softmax function (makes it differentiable)
        scaling=10.**(np.floor(np.log10(mass))-1) #make it so the exponentials are taking ~10 (closer numerically, while still differentiable)
        mass=scaling*np.log(np.exp(energy_mass/scaling)+np.exp(power_mass/scaling))  #write it this way to prevent overflow

    battery.mass_properties.mass=mass
    battery.max_energy=battery.specific_energy*mass
    battery.max_power =battery.specific_power*mass
    
    return