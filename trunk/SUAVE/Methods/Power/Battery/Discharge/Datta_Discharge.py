"""models discharge losses based on an empirical correlation"""
#by M. Vegh
#Based on method taken from Datta and Johnson: 
#"Requirements for a Hydrogen Powered All-ElectricManned Helicopter"
""" SUAVE Methods for Energy Systems """

# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------

import numpy as np

# ----------------------------------------------------------------------
#  Methods
# ----------------------------------------------------------------------

def datta_discharge(battery,numerics): #adds a battery that is optimized based on power and energy requirements and technology
    Ibat  = battery.inputs.current
    pbat  = battery.inputs.power_in
    edraw = battery.inputs.energy_transfer
    Rbat  = battery.resistance
    I     = numerics.integrate_time
    
    # Maximum energy
    max_energy = battery.max_energy
    
    #state of charge of the battery
    x = np.divide(battery.current_energy,battery.max_energy())[:,0,None]

    # C rate from 
    C = 3600.*pbat/battery.max_energy()
    
    # Empirical value for discharge
    x[x<-35.] = -35. # Fix x so it doesn't warn
    
    f = 1-np.exp(-20*x)-np.exp(-20*(1-x)) 
    
    f[x<0.0] = 0.0 # Negative f's don't make sense
    
    # Model discharge characteristics based on changing resistance
    R = Rbat*(1+C*f)
    
    # Calculate resistive losses
    Ploss = (Ibat**2)*R
    
    # Energy loss from power draw
    eloss = np.dot(I,Ploss)
    

    # Pack up
    battery.current_energy=battery.current_energy[0]*np.ones_like(eloss)
   

    delta = 0.0
    flag  = 0
    battery.current_energy = battery.current_energy[0] * np.ones_like(eloss) 
    for ii in range(1,len(edraw)):
        if (edraw[ii,0] > (max_energy- battery.current_energy[ii-1])):
            flag = 1 
            delta = delta + ((max_energy- battery.current_energy[ii-1]) - edraw[ii,0] + np.abs(eloss[ii]))
            edraw[ii,0] = edraw[ii,0] + delta
        elif flag ==1:
            edraw[ii,0] = edraw[ii,0] + delta
        battery.current_energy[ii] = battery.current_energy[ii] + edraw[ii] - np.abs(eloss[ii])
                
    return