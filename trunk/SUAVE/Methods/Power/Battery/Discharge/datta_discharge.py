## @ingroup Methods-Power-Battery-Discharge
# datta_discharge.py
# 
# Created:  ### ####, M. Vegh
# Modified: Feb 2016, E. Botero

# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------

import numpy as np

# ----------------------------------------------------------------------
#  Datta Discharge
# ----------------------------------------------------------------------

## @ingroup Methods-Power-Battery-Discharge
def datta_discharge(battery,numerics): 
    """models discharge losses based on an empirical correlation
       Based on method taken from Datta and Johnson: 
       
       Assumptions: 
       Constant Peukart coefficient
       
       Source:
       "Requirements for a Hydrogen Powered All-Electric Manned Helicopter" by Datta and Johnson
      
       voltage model from from Chen, M. and Rincon-Mora, G. A., "Accurate Electrical Battery Model Capable of Predicting
       # Runtime and I - V Performance" IEEE Transactions on Energy Conversion, Vol. 21, No. 2, June 2006, pp. 504-511
       
       Inputs:
       battery.
        resistance                      [Ohms]
        max_energy                      [Joules]
        current_energy (to be modified) [Joules]
        inputs.
            current                     [amps]
            power_in                    [Watts]
       
       Outputs:
       battery.
        current energy                  [Joules]
        resistive_losses                [Watts]
        voltage_open_circuit            [Volts]
        voltage_under_load              [Volts]
        
        
    """
    
    Ibat  = battery.inputs.current
    pbat  = battery.inputs.power_in
    Rbat  = battery.resistance
    v_max = battery.max_voltage
    I     = numerics.time.integrate
    D     = numerics.time.differentiate

    # Maximum energy
    max_energy = battery.max_energy
    
    #state of charge of the battery
    initial_discharge_state = np.dot(I,pbat) + battery.current_energy[0]
    x = np.divide(initial_discharge_state,battery.max_energy)

    # C rate
    C = np.abs(3600.*pbat/battery.max_energy)
    
    # Empirical value for discharge
    x[x<-35.] = -35. # Fix x so it doesn't warn
    
    f = 1-np.exp(-20.*x)-np.exp(-20.*(1.-x))
    
    f[f<0.0] = 0.0 # Negative f's don't make sense
    f = np.reshape(f, np.shape(C))
    
    # Model discharge characteristics based on changing resistance
    R          = Rbat*(1.+np.multiply(C,f))
    R[R==Rbat] = 0.  #when battery isn't being called
    
    # Calculate resistive losses
    Ploss = (Ibat**2.)*R
    
    # Power going into the battery accounting for resistance losses
    P = pbat - np.abs(Ploss)
    
    # Possible Energy going into the battery:
    energy_unmodified = np.dot(I,P)
    
    # Available capacity
    capacity_available = max_energy - battery.current_energy[0]
   
    # How much energy the battery could be overcharged by
    delta           = energy_unmodified -capacity_available
    delta[delta<0.] = 0.
    
    # Power that shouldn't go in
    ddelta = np.dot(D,delta) 
    
    # Power actually going into the battery
    P[P>0.] = P[P>0.] - ddelta[P>0.]
    ebat = np.dot(I,P)
    ebat = np.reshape(ebat,np.shape(battery.current_energy)) #make sure it's consistent
    
    # Add this to the current state
    if np.isnan(ebat).any():
        ebat=np.ones_like(ebat)*np.max(ebat)
        if np.isnan(ebat.any()): #all nans; handle this instance
            ebat=np.zeros_like(ebat)
            
    current_energy = ebat + battery.current_energy[0]
    
    new_x = np.divide(current_energy,battery.max_energy)
            
    # A voltage model from Chen, M. and Rincon-Mora, G. A., "Accurate Electrical Battery Model Capable of Predicting
    # Runtime and I - V Performance" IEEE Transactions on Energy Conversion, Vol. 21, No. 2, June 2006, pp. 504-511
    v_normalized         = (-1.031*np.exp(-35.*new_x) + 3.685 + 0.2156*new_x - 0.1178*(new_x**2.) + 0.3201*(new_x**3.))/4.1
    voltage_open_circuit = v_normalized * v_max
    
    # Voltage under load:
    voltage_under_load   = voltage_open_circuit  - Ibat*R
        
    # Pack outputs
    battery.current_energy       = current_energy
    battery.resistive_losses     = Ploss
    battery.voltage_open_circuit = voltage_open_circuit
    battery.voltage_under_load   = voltage_under_load
    battery.state_of_charge      = new_x
    
    return
