## @ingroup Methods-Missions-Segments-Ground
# Idle.py

# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------
import numpy as np 
# ----------------------------------------------------------------------
#  Initialize Conditions
# ----------------------------------------------------------------------  

## @ingroup Methods-Missions-Segments-Ground
def unpack_unknowns(segment): 
    
    pass

# ----------------------------------------------------------------------
#  Initialize Conditions
# ----------------------------------------------------------------------

## @ingroup Methods-Missions-Segments-Ground 
def initialize_conditions(segment): 
    
    # unpack   
    if segment.state.initials:
        energy_initial = segment.state.initials.conditions.propulsion.battery_energy[-1,0]  
    elif 'battery_energy' in segment:
        energy_initial = segment.battery_energy  
    else:
        energy_initial = 0.0
        
    duration   = segment.time 
    if segment.battery_discharge == False: 
        delta_energy = segment.max_energy*segment.charging_SOC_cutoff - energy_initial
        duration = delta_energy/(segment.charging_current*segment.charging_voltage) 
        
    t_nondim   = segment.state.numerics.dimensionless.control_points
    conditions = segment.state.conditions   
    
    # dimensionalize time
    t_initial = conditions.frames.inertial.time[0,0]
    t_nondim  = segment.state.numerics.dimensionless.control_points
    time      =  t_nondim * (duration) + t_initial
    
    segment.state.conditions.frames.inertial.time[:,0] = time[:,0] 
    

def update_battery_age(segment):  
     
    SOC     = segment.conditions.propulsion.state_of_charge
    V_oc    = segment.conditions.propulsion.voltage_open_circuit
    t       = segment.conditions.propulsion.battery_age_in_days 
    Q_prior = segment.conditions.propulsion.battery_charge_throughput
    
    # aging model  
    delta_DOD = abs(SOC[0][0] - SOC[-1][0])
    rms_V_oc = np.sqrt(np.mean(V_oc**2)) 
    alpha_cap = 0*((7.542*V_oc[0][0] - 23.75)*1E6) * np.exp(-6976/(V_oc[0][0]+273))
    alpha_res = 0*((5.270*V_oc[0][0] - 16.32)*1E5) * np.exp(-5986/(V_oc[0][0]+273))
    beta_cap  = 7.348E-3 * (rms_V_oc - 3.667)**2 +  7.60E-4 + 4.081E-3*delta_DOD
    beta_res  = 2.153E-4 * (rms_V_oc - 3.725)**2 - 1.521E-5 + 2.798E-4*delta_DOD
     
    segment.conditions.propulsion.battery_capacity_growth_factor   = (1 - alpha_cap*(t**0.75) - beta_cap*np.sqrt(Q_prior))  
    segment.conditions.propulsion.battery_resistance_growth_factor = (1 + alpha_res*(t**0.75) + beta_res*Q_prior) 
        
        
        
