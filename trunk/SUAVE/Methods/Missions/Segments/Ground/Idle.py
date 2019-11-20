## @ingroup Methods-Missions-Segments-Ground
# Idle.py

# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------
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
        energy_initial            = segment.state.initials.conditions.propulsion.battery_energy[-1,0]  
    elif 'battery_energy' in segment:
        energy_initial            = segment.battery_energy  
    else:
        energy_initial            = 0.0
        
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
    
    segment.state.conditions.frames.inertial.time[:,0]            = time[:,0]      
