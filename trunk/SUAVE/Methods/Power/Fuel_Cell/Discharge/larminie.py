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

def larminie(fuel_cell,conditions,numerics): #adds a battery that is optimized based on power and energy requirements and technology
    A             = fuel_cell.interface_area/(Units.cm**2.)
    r             = fuel_cell.r/(Units.kohm/(Units.cm**2))
    Eoc           = fuel_cell.Eoc 
    A1            = fuel_cell.A1  
    m             = fuel_cell.m   
    n             = fuel_cell.n   
    
    
    
    
    power         = fuel_cell.inputs.power_in  
    i1            = np.linspace(.1,1200.0,200.0)*Units.mA*Units.cm**2                                                     #current density(mA cm^-2): use vector of these with interpolation to find values
    v             = Eoc-r*i1-A1*np.log(i1)-m*np.exp(n*i1)                                                                 #useful voltage vector
    efficiency    = np.divide(v,1.48)                                                                                     #efficiency of the cell vs voltage
    p             = fuel_cell.number_of_cells* np.divide(np.multiply(v,i1),1000.0)*A                                      #obtain power output in W
    imax          = np.argmax(p)
    
    if power.any()>p[imax]:            
        print "Warning, maximum power output of fuel cell exceeded"
    
    p             = np.resize(p,imax+1)                                                                                  #resize vector such that it only goes to max power to prevent ambiguity
    print np.len(power)
    print np.len(efficiency)
    mdot_vec      = np.divide(power,np.multiply(fuel_cell.propellant.specific_energy,efficiency))                        #mass flow rate of the fuel based on fuel cell efficiency              
    ip            = np.argmin(np.abs(p-power))                                                                           #find index such that that operating power is equal to required power 
    v1            = v[ip]                                                                                                #operating voltage of a single stack    
    efficiency_out= efficiency[ip] 
    mdot          = np.divide(power,np.multiply(fuel_cell.propellant.specific_energy,efficiency_out))                    #mass flow rate of hydrogen

    return mdot