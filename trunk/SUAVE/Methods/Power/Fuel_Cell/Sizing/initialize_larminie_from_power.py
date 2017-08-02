## @ingroup Methods-Power-Fuel_Cell-Sizing
# initialize_larminie_from_power.py
#
# Created : Apr 2015, M. Vegh 
# Modified: Sep 2015, M. Vegh
#           Feb 2016, E. Botero

# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------

import scipy as sp
import numpy as np
from SUAVE.Core import Units
from SUAVE.Methods.Power.Fuel_Cell.Discharge.find_power_larminie import find_power_larminie

# ----------------------------------------------------------------------
#  Initialize Larminie from Power
# ----------------------------------------------------------------------

## @ingroup Methods-Power-Fuel_Cell-Sizing
def initialize_larminie_from_power(fuel_cell,power): 
    '''
    Initializes extra paramters for the fuel cell when using the larminie method
    Determines the number of stacks
    
    Inputs:
    power                 [W]
    fuel_cell
    
    Outputs:
    
    fuel_cell.
        power_per_cell    [W]
        number_of_cells
        max_power         [W]
        volume            [m**3]
        specific_power    [W/kg]
        mass_properties.
            mass          [kg]
       
        
    '''
    
    
    
    fc                      = fuel_cell
    lb                      = .1*Units.mA/(Units.cm**2.)    #lower bound on fuel cell current density
    ub                      = 1200.0*Units.mA/(Units.cm**2.)
    sign                    = -1. #used to minimize -power
    current_density         = sp.optimize.fminbound(find_power_larminie, lb, ub, args=(fc, sign))
    power_per_cell          = find_power_larminie(current_density,fc)
    
    fc.number_of_cells      = np.ceil(power/power_per_cell)
    fc.max_power            = fc.number_of_cells*power_per_cell
    fc.volume               = fc.number_of_cells*fc.interface_area*fc.wall_thickness
    fc.mass_properties.mass = fc.volume*fc.cell_density*fc.porosity_coefficient #fuel cell mass in kg
    fc.mass_density         = fc.mass_properties.mass/  fc.volume                      
    fc.specific_power       = fc.max_power/fc.mass_properties.mass #fuel cell specific power in W/kg    