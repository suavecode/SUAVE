#Created by M. Vegh 4/23/15

""" Calculates mass flow of fuel cell based on method from Larminie and 
Dicks (Fuel Cell Systems Explained) """

# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------

import numpy as np

# ----------------------------------------------------------------------
#  Methods
# ----------------------------------------------------------------------

def larminie(fuel_cell,conditions,numerics): #adds a battery that is optimized based on power and energy requirements and technology
    power         = fuel_cell.inputs.power_in
    i1            = np.linspace(.1,1200.0,200.0)                                                   #current density(mA cm^-2): use vector of these with interpolation to find values
    v             =fuel_cell.Eoc-fuel_cell.r*i1-fuel_cell.A1*np.log(i1)-fuel_cell.m*np.exp(fuel_cell.n*i1)                  #useful voltage vector
    efficiency    =np.divide(v,1.48)                                                      #efficiency of the cell vs voltage
    p             =fuel_cell.Ncell* np.divide(np.multiply(v,i1),1000.0)*fuel_cell.A                          #obtain power output in W
    imax          =np.argmax(p)
    
    if power.any()>p[imax]:            
        print "Warning, maximum power output of fuel cell exceeded"
    
    p             =np.resize(p,imax+1)                                                             #resize vector such that it only goes to max power to prevent ambiguity
    mdot_vec      =np.divide(power,np.multiply(fuel_cell.propellant.specific_energy,efficiency)) #mass flow rate of the fuel based on fuel cell efficiency              
    ip            =np.argmin(np.abs(p-power))                                                     #find index such that that operating power is equal to required power 
    v1            =v[ip]                                                                          #operating voltage of a single stack    
    efficiency_out=efficiency[ip] 
    mdot          =np.divide(power,np.multiply(fuel_cell.propellant.specific_energy,efficiency_out))  #mass flow rate of hydrogen

    return [mdot]