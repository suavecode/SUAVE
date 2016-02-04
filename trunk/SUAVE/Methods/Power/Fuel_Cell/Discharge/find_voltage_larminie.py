# find_voltage_larminie.py
#
# Created : Apr 2015, M. Vegh 
# Modified: Feb 2016, E. Botero
  
# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------

import autograd.numpy as np 
from SUAVE.Core import Units

# ----------------------------------------------------------------------
#  Find Voltage Larminie
# ----------------------------------------------------------------------

def find_voltage_larminie(fuel_cell,current_density):
    
    r   = fuel_cell.r/(Units.kohm*(Units.cm**2))
    Eoc = fuel_cell.Eoc 
    A1  = fuel_cell.A1  
    m   = fuel_cell.m   
    n   = fuel_cell.n   
    
    i1 = current_density/(Units.mA/(Units.cm**2.)) #current density(mA cm^-2)
    v  = Eoc-r*i1-A1*np.log(i1)-m*np.exp(n*i1)     #useful voltage vector

    return v