## @ingroup Methods-Missions-Segments-Battery_Cell_Testbench
# Charge_Discharge_Test.py

# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------
import numpy as np 
# ----------------------------------------------------------------------
#  Initialize Conditions
# ----------------------------------------------------------------------  

## @ingroup Methods-Missions-Segments-Battery_Cell_Testbench
def unpack_unknowns(segment): 
    
    pass

# ----------------------------------------------------------------------
#  Initialize Conditions
# ----------------------------------------------------------------------

## @ingroup Methods-Missions-Segments-Battery_Cell_Testbench 
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
        E_growth_factor = segment.conditions.propulsion.battery_capacity_fade_factor
        delta_energy    = segment.max_energy*E_growth_factor*segment.charging_SOC_cutoff - energy_initial
        duration        = delta_energy*1.25/(segment.charging_current*segment.charging_voltage) 
        
    t_nondim   = segment.state.numerics.dimensionless.control_points
    conditions = segment.state.conditions   
    
    # dimensionalize time
    t_initial = conditions.frames.inertial.time[0,0]
    t_nondim  = segment.state.numerics.dimensionless.control_points
    time      =  t_nondim * (duration) + t_initial
    
    segment.state.conditions.frames.inertial.time[:,0] = time[:,0] 
    
