#Created by M. Vegh 4/23/15

""" Calculates mass flow of fuel cell based solely on specific energy
end base efficiency factor """

# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------

import numpy as np

# ----------------------------------------------------------------------
#  Methods
# ----------------------------------------------------------------------

def initialize_from_power(fuel_cell,power): #adds a fuel cell that is sized based on the specific power of the fuel cell
    fuel_cell.mass_properties.mass=power/fuel_cell.specific_power
    return