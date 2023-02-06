## @ingroup Methods-Power-Battery 
# append_initial_battery_conditions.py
# 
# Created:  Sep 2021, M. Clarke 
# Modified: Oct 2021, E. Botero

# ----------------------------------------------------------------------
#  Imports
# ---------------------------------------------------------------------- 
import MARC
from MARC.Core import Data

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
                 battery.cycle_in_day               [unitless]
                 battery.pack.temperature           [Kelvin]
                 battery.charge_throughput          [Ampere-Hours] 
                 battery.resistance_growth_factor   [unitless]
                 battery.capacity_fade_factor       [unitless]
                 battery.discharge                  [boolean]
                 increment_battery_cycle_day        [boolean]
               
        Outputs:
            segment
               battery_discharge                    [boolean]
               increment_battery_cycle_day          [boolean]
            segment.state.conditions.propulsion
               battery.discharge_flag               [boolean]
               battery.pack.max_initial_energy      [watts]
               battery.pack.energy                  [watts]
               battery.pack.max_aged_energy         [watts]    
               battery.pack.temperature             [kelvin]
               battery.cycle_in_day                 [int]
               battery.cell.charge_throughput       [Ampere-Hours] 
               battery.resistance_growth_factor     [unitless]
               battery.capacity_fade_factor         [unitless]



    
        Properties Used:
        None
    """      
    # unpack
    propulsion = segment.state.conditions.propulsion
    
    # compute ambient conditions
    atmosphere    = MARC.Analyses.Atmospheric.US_Standard_1976()
    alt           = -segment.conditions.frames.inertial.position_vector[:,2] 
    if segment.temperature_deviation != None:
        temp_dev = segment.temperature_deviation    
    atmo_data  = atmosphere.compute_values(altitude = alt,temperature_deviation=temp_dev)
    
    
    # Set if it is a discharge segment
    if 'battery_discharge' not in segment:      
        segment.battery_discharge          = True
        propulsion.battery.discharge_flag  = True
    else:
        propulsion.battery.discharge_flag  = segment.battery_discharge
        
        
    # This is the only one besides energy and discharge flag that should be packed into the segment top level
    if 'increment_battery_cycle_day' not in segment:
        segment.increment_battery_cycle_day   = False    
        
    # If an initial segment with battery energy set 
    if 'battery_pack_temperature' not in segment:     
        pack_temperature              = atmo_data.temperature[0,0]
    else:
        pack_temperature              = segment.battery_pack_temperature 
    propulsion.battery.pack.temperature[:,0] = pack_temperature
    propulsion.battery.cell.temperature[:,0] = pack_temperature
    
    
    if 'battery_max_aged_energy' in segment:
        battery_max_aged_energy = segment.battery_max_aged_energy
    else:
        battery_max_aged_energy = battery.pack.max_energy
        
        
    propulsion.battery.pack.max_aged_energy = battery_max_aged_energy   
    
    
        
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
        propulsion.battery.pack.max_initial_energy           = initial_mission_energy
        propulsion.battery.pack.energy[:,0]                  = initial_segment_energy 
        propulsion.battery.cell.cycle_in_day                 = cycle_day        
        propulsion.battery.cell.charge_throughput[:,0]       = cell_charge_throughput 
        propulsion.battery.cell.resistance_growth_factor     = resistance_growth_factor 
        propulsion.battery.cell.capacity_fade_factor         = capacity_fade_factor
        propulsion.battery.cell.state_of_charge[:,0]         = initial_mission_energy/battery_max_aged_energy
            
    return 
    
    
