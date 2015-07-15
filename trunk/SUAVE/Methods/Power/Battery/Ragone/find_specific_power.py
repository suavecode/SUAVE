'''determines specific energy from a ragone curve correlation'''
#by M. Vegh

""" SUAVE Methods for Energy Systems """

# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------

import numpy as np

# ----------------------------------------------------------------------
#  Methods
# ----------------------------------------------------------------------

def find_specific_power(battery, specific_energy): #adds a battery that is optimized based on power and energy requirements and technology
    const_1=battery.ragone.const_1
    const_2=battery.ragone.const_2
    specific_power=const_1*10**(const_2*specific_energy)
    battery.specific_power =specific_power
    battery.specific_energy=specific_energy
    
    return