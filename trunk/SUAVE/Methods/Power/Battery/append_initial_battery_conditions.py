## @ingroup Methods-Power-Battery 
# append_initial_battery_conditions.py
# 
# Created:  Sep 2021, M. Clarke 
# Modified: Oct 2021, E. Botero

# ----------------------------------------------------------------------
#  Imports
# ---------------------------------------------------------------------- 
import SUAVE
from SUAVE.Core import Data

# ----------------------------------------------------------------------
#  Methods
# ----------------------------------------------------------------------
## @ingroup Methods-Power-Battery 
def append_initial_battery_conditions(segment,battery): 
    """ Packs the initial battery conditions
    
        Assumptions:
        Battery temperature is set to one degree hotter than ambient 
        temperature for robust convergence. Initial mission energy, maxed aged energy, and 
        initial segment energy are the same. Cycle day is zero unless specified, resistance_growth_factor and
        capacity_fade_factor is one unless specified in the segment
    
        Source:
        N/A
    
        Inputs:  
            atmosphere.temperature             [Kelvin]
            
            Optional:
            segment.
                 battery_cycle_day                  [unitless]
                 battery_pack_temperature           [Kelvin]
                 battery_charge_throughput          [Ampere-Hours] 
                 battery_resistance_growth_factor   [unitless]
                 battery_capacity_fade_factor       [unitless]
                 battery_discharge                  [boolean]
                 increment_battery_cycle_day        [boolean]
               
        Outputs:
            segment
               battery_discharge                    [boolean]
               increment_battery_cycle_day          [boolean]
            segment.state.conditions.propulsion
               battery_discharge_flag               [boolean]
               battery_max_initial_energy           [watts]
               battery_energy                       [watts]
               battery_max_aged_energy              [watts]    
               battery_pack_temperature             [kelvin]
               battery_cycle_day                    [int]
               battery_cell_charge_throughput       [Ampere-Hours] 
               battery_resistance_growth_factor     [unitless]
               battery_capacity_fade_factor         [unitless]



    
        Properties Used:
        None
    """      
    # unpack
    propulsion = segment.state.conditions.propulsion
    
    # compute ambient conditions
    atmosphere    = SUAVE.Analyses.Atmospheric.US_Standard_1976()
    alt           = -segment.conditions.frames.inertial.position_vector[:,2] 
    if segment.temperature_deviation != None:
        temp_dev = segment.temperature_deviation    
    atmo_data  = atmosphere.compute_values(altitude = alt,temperature_deviation=temp_dev)
    
    
    # Set if it is a discharge segment
    if 'battery_discharge' not in segment:      
        segment.battery_discharge          = True
        propulsion.battery_discharge_flag  = True
    else:
        propulsion.battery_discharge_flag  = segment.battery_discharge
        
        
    # This is the only one besides energy and discharge flag that should be packed into the segment top level
    if 'increment_battery_cycle_day' not in segment:
        segment.increment_battery_cycle_day   = False    
        
    # If an initial segment with battery energy set 
    if 'battery_pack_temperature' not in segment:     
        pack_temperature              = atmo_data.temperature[0,0]
    else:
        pack_temperature              = segment.battery_pack_temperature 
    propulsion.battery_pack_temperature[:,0] = pack_temperature
    propulsion.battery_cell_temperature[:,0] = pack_temperature
    
    
    if 'battery_max_aged_energy' in segment:
        battery_max_aged_energy = segment.battery_max_aged_energy
    else:
        battery_max_aged_energy = battery.max_energy
        
        
    propulsion.battery_max_aged_energy = battery_max_aged_energy   
    
    
        
    if 'battery_energy' in segment: 
        
        initial_segment_energy         = segment.battery_energy
        initial_mission_energy         = segment.battery_energy 
        
        if 'battery_cycle_day' not in segment: 
            cycle_day                     = 0
        else:
            cycle_day                     = segment.battery_cycle_day

        if 'battery_cell_charge_throughput' not in segment:
            cell_charge_throughput        = 0.
        else:
            cell_charge_throughput        = segment.battery_cell_charge_throughput
            
        if 'battery_resistance_growth_factor' not in segment:
            resistance_growth_factor      = 1 
        else:
            resistance_growth_factor      = segment.battery_resistance_growth_factor
            
        if 'battery_capacity_fade_factor' not in segment: 
            capacity_fade_factor          = 1   
        else:
            capacity_fade_factor          = segment.battery_capacity_fade_factor
        
        # Pack into conditions
        propulsion.battery_max_initial_energy           = initial_mission_energy
        propulsion.battery_energy[:,0]                  = initial_segment_energy 

        propulsion.battery_cycle_day                    = cycle_day        
        propulsion.battery_cell_charge_throughput[:,0]  = cell_charge_throughput 
        propulsion.battery_resistance_growth_factor     = resistance_growth_factor 
        propulsion.battery_capacity_fade_factor         = capacity_fade_factor
        propulsion.battery_state_of_charge[:,0]         = initial_mission_energy/battery_max_aged_energy
            
    return 
    
    
