"""finds the mass gain rate of the battery from the ambient air"""
#by M. Vegh

""" SUAVE Methods for Energy Systems """

# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------

import numpy as np

# ----------------------------------------------------------------------
#  Methods
# ----------------------------------------------------------------------

def find_total_mass_gain(battery): #adds a battery that is optimized based on power and energy requirements and technology
    mgain=battery.max_energy*battery.mass_gain_factor
    return mgain