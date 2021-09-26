## @ingroup Methods-Power-Battery 
# append_initial_battery_conditions.py
# 
# Created: Sep 2021, M. Clarke 
import SUAVE

# ----------------------------------------------------------------------
#  Methods
# ----------------------------------------------------------------------
## @ingroup Methods-Power-Battery 
def append_initial_battery_conditions(segment,initial_battery_cell_thevenin_voltage = 0.1): 
    """ Packs the initial battery conditions of the first segment
    
        Assumptions:
        Battery temperature is set to one degree hotter than ambient 
        temperature for robust convergence 
    
        Source:
        N/A
    
        Inputs:  
               atmosphere.temperature             [Kelvin]
               
        Outputs:
            segment.
               battery_cycle_day [days]
               battery_pack_temperature           [Kelvin]
               battery_charge_throughput          [Ampere-Hours] 
               battery_resistance_growth_factor   [unitless]
               battery_capacity_fade_factor       [unitless]
               battery_thevenin_voltage           [Volts]  
    
        Properties Used:
        None
    """      
    # compute ambient conditions
    atmosphere    = SUAVE.Analyses.Atmospheric.US_Standard_1976()
    alt           = -segment.conditions.frames.inertial.position_vector[:,2] 
    if segment.temperature_deviation != None:
        temp_dev = segment.temperature_deviation    
    atmo_data  = atmosphere.compute_values(altitude = alt,temperature_deviation=temp_dev + 1.)  
    
    if 'battery_cycle_day' not in segment: 
        segment.battery_cycle_day = 0   
        
    if 'battery_pack_temperature' not in segment:     
        segment.battery_pack_temperature  = atmo_data.temperature[0,0]
        
    if 'battery_charge_throughput' not in segment:
        segment.battery_charge_throughput = 0   
    
    if 'battery_resistance_growth_factor' not in segment:
        segment.battery_resistance_growth_factor = 1 
        
    if 'battery_capacity_fade_factor' not in segment: 
        segment.battery_capacity_fade_factor = 1   
    
    if 'battery_thevenin_voltage' not in segment: 
        segment.battery_thevenin_voltage  = initial_battery_cell_thevenin_voltage  
        
    if 'battery_discharge' not in segment:      
        segment.battery_discharge = True          
            
    return 
    
    
