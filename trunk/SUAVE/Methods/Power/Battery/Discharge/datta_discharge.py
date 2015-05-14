"""models discharge losses based on an empirical correlation"""
#by M. Vegh
#Based on method taken from Datta and Johnson: 
#"Requirements for a Hydrogen Powered All-Electric Manned Helicopter"
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
    Rbat  = battery.resistance
    I     = numerics.time.integrate
    D     = numerics.time.differentiate
    
    # Maximum energy
    max_energy = battery.max_energy
    
    #state of charge of the battery
   
    x = np.divide(battery.current_energy,battery.max_energy)

    # C rate from 
    C = 3600.*pbat/battery.max_energy
    
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

    # Possible Energy going into the battery:
    energy_unmodified = np.dot(I,pbat-Ploss)
    
    # Available capacity
    capacity_available = max_energy - battery.current_energy[0]
   
    # How much energy the battery could be overcharged by
    delta = energy_unmodified -capacity_available
    delta[delta<0.] = 0.
    
    ddelta = np.dot(D,delta) # Power that shouldn't go in
    
    # Power actually going into the battery
    P = pbat - Ploss
    P[P>0.] = P[P>0.] - ddelta[P>0.]
    ebat = np.dot(I,P)
    
    # Add this to the current state
    battery.current_energy   = ebat + battery.current_energy[0]
    battery.resistive_losses = Ploss
    
    return