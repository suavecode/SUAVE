#Created by M. Vegh 4/27/15

""" sets up additional values of fuel cell to run method from Larminie and 
Dicks (Fuel Cell Systems Explained) """

# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------

import numpy as np
import SUAVE
from SUAVE.Core import Units
from larminie import larminie
# ----------------------------------------------------------------------
#  Methods
# ----------------------------------------------------------------------

def setup_larminie(fuel_cell):                        #default values representative of a hydrogen fuel cell
   fuel_cell.number_of_cells       = 0.0                                  #number of fuel cells in the stack
   fuel_cell.interface_area        = 875.*(Units.cm**2.)                  # area of the fuel cell interface
   fuel_cell.r                     = (2.45E-4) *(Units.kohm*(Units.cm**2))# area specific resistance [k-Ohm-cm^2]
   fuel_cell.Eoc                   = .931                                 # effective activation energy (V)
   fuel_cell.A1                    = .03                                  # slope of the Tafel line (models activation losses) (V)
   fuel_cell.m                     = 1.05E-4                              # constant in mass-transfer overvoltage equation (V)
   fuel_cell.n                     = 8E-3                                 # constant in mass-transfer overvoltage equation
   fuel_cell.ideal_voltage         = 1.48
   fuel_cell.wall_thickness        = .0022224                             # thickness of cell wall in meters  
   fuel_cell.cell_density          =1988.                                 # cell density in kg/m^3
   fuel_cell.porosity_coefficient  =.6                                    # porosity coefficient
   fuel_cell.discharge_model       = larminie
   return