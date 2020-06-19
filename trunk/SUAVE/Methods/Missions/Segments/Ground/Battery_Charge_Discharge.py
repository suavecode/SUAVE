## @ingroup Methods-Missions-Segments-Ground
# Battery_Charge_Discharge.py

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
        intial_segment_energy = segment.state.initials.conditions.propulsion.battery_energy[-1,0]  
    elif 'battery_energy' in segment:
        intial_segment_energy = segment.battery_energy  
    else:
        intial_segment_energy = 0.0
        
    duration   = segment.time 
    if segment.battery_discharge == False: 
        E_growth_factor = segment.conditions.propulsion.battery_capacity_fade_factor
        delta_energy    = segment.max_energy*E_growth_factor*segment.charging_SOC_cutoff - intial_segment_energy
        duration        = delta_energy*1.25/(segment.charging_current*segment.charging_voltage) 
        
    t_nondim   = segment.state.numerics.dimensionless.control_points
    conditions = segment.state.conditions   
    
    # dimensionalize time
    t_initial = conditions.frames.inertial.time[0,0]
    t_nondim  = segment.state.numerics.dimensionless.control_points
    time      =  t_nondim * (duration) + t_initial
    
    segment.state.conditions.frames.inertial.time[:,0] = time[:,0] 
    
