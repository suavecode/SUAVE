#initialize_larminie_from_power
#Created: M. Vegh, 4/23/15
#Modified:M. Vegh, September 2015
""" adds a fuel cell that is sized based on the specific power of the fuel cell,
uses method from Larminie and Dicks (Fuel Cell Systems Explained)
 to estimate power
 """

# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------
import scipy as sp
import numpy as np
import SUAVE
from SUAVE.Core import Units
from SUAVE.Methods.Power.Fuel_Cell.Discharge.find_power_larminie import find_power_larminie
# ----------------------------------------------------------------------
#  Methods
# ----------------------------------------------------------------------

def initialize_larminie_from_power(fuel_cell,power): 
    lb                            =.1*Units.mA/(Units.cm**2.)    #lower bound on fuel cell current density
    ub                            =1200.0*Units.mA/(Units.cm**2.)
    sign                          =-1. #used to minimize -power
    current_density               =sp.optimize.fminbound(find_power_larminie, lb, ub, args=(fuel_cell, sign))
    power_per_cell                =find_power_larminie(current_density, fuel_cell)
    fuel_cell.number_of_cells     =np.ceil(power/power_per_cell)
    fuel_cell.max_power           =fuel_cell.number_of_cells*power_per_cell
    fuel_cell.volume              =fuel_cell.number_of_cells*fuel_cell.interface_area*fuel_cell.wall_thickness
    fuel_cell.mass_properties.mass=fuel_cell.volume*fuel_cell.cell_density*fuel_cell.porosity_coefficient       #fuel cell mass in kg
    fuel_cell.mass_density        = fuel_cell.mass_properties.mass/  fuel_cell.volume                      
    fuel_cell.specific_power      =fuel_cell.max_power/fuel_cell.mass_properties.mass                     #fuel cell specific power in W/kg    

    return