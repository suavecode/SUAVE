#Created by M. Vegh 4/23/15

""" Calculates mass flow of fuel cell based on method from Larminie and 
Dicks (Fuel Cell Systems Explained) """

# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------

import numpy as np
import SUAVE
from SUAVE.Core import Units
# ----------------------------------------------------------------------
#  Methods
# ----------------------------------------------------------------------

def find_voltage_larminie(fuel_cell,current_density): #adds a battery that is optimized based on power and energy requirements and technology
    r             = fuel_cell.r/(Units.kohm*(Units.cm**2))
    Eoc           = fuel_cell.Eoc 
    A1            = fuel_cell.A1  
    m             = fuel_cell.m   
    n             = fuel_cell.n   
    i1            = current_density/(Units.mA/(Units.cm**2.))        #current density(mA cm^-2)
    v             = Eoc-r*i1-A1*np.log(i1)-m*np.exp(n*i1)                                                                 #useful voltage vector
                                                                                   #efficiency of the cell vs voltage
    
    return v