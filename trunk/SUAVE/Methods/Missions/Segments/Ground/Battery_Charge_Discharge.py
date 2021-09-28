## @ingroup Methods-Missions-Segments-Ground
# Battery_Charge_Discharge.py
#
# Created: Aug. 2021, M. Clarke

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
    overcharge_contingency = segment.overcharge_contingency = 1.25
    
    # unpack   
    if segment.state.initials:
        intial_segment_energy = segment.state.initials.conditions.propulsion.battery_energy[-1,0]  
        segment_max_energy    = segment.state.initials.conditions.propulsion.battery_max_aged_energy
    elif 'battery_energy' in segment:
        intial_segment_energy = segment.battery_energy  
        segment_max_energy    = segment.battery_energy
    
    duration  = segment.time  
    if segment.battery_discharge == False: 
        for network in segment.analyses.energy.network:  
            battery        = network.battery
            charge_current = battery.cell.charging_current * battery.pack_config.parallel
            charge_voltage = battery.cell.charging_voltage * battery.pack_config.series
            delta_energy   = segment_max_energy - intial_segment_energy
            duration       = delta_energy*overcharge_contingency/(charge_current*charge_voltage) 
        
    t_nondim   = segment.state.numerics.dimensionless.control_points
    conditions = segment.state.conditions   
    
    # dimensionalize time
    t_initial = conditions.frames.inertial.time[0,0]
    t_nondim  = segment.state.numerics.dimensionless.control_points
    time      =  t_nondim * (duration) + t_initial
    
    segment.state.conditions.frames.inertial.time[:,0] = time[:,0] 
    
