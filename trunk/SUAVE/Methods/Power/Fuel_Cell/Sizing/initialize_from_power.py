# initialize_from_power.py
#
# Created : Apr 2015, M. Vegh 
# Modified: Feb 2016, E. Botero
           
# ----------------------------------------------------------------------
#  Initialize from Power
# ----------------------------------------------------------------------

def initialize_from_power(fuel_cell,power):
    
    fuel_cell.mass_properties.mass=power/fuel_cell.specific_power
