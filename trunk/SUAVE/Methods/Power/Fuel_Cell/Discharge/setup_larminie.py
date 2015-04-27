#Created by M. Vegh 4/27/15

""" sets up additional values of fuel cell to run method from Larminie and 
Dicks (Fuel Cell Systems Explained) """

# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------

import numpy as np

# ----------------------------------------------------------------------
#  Methods
# ----------------------------------------------------------------------

def setup_larminie(fuel_cell)                        #default values representative of a hydrogen fuel cell
   self.Ncell = 0.0                                  #number of fuel cells in the stack
   self.A     = 875*(Units.cm**2)                    # area of the fuel cell interface
   self.r     = (2.45E-4) *(Units.kOhm/(Units.cm**2))# area specific resistance [k-Ohm-cm^2]
   self.Eoc   = .931                                 # effective activation energy (V)
   self.A1    = .03                                  # slope of the Tafel line (models activation losses) (V)
   self.m     = 1.05E-4                              # constant in mass-transfer overvoltage equation (V)
   self.n     = 8E-3                                 # constant in mass-transfer overvoltage equation
   
   return