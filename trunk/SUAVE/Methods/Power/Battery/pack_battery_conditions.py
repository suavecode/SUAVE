## @ingroup Methods-Power-Battery 
# pack_battery_conditions.py
# 
# Created: Sep 2021, M. Clarke 

# ----------------------------------------------------------------------
#  Methods
# ----------------------------------------------------------------------
## @ingroup Methods-Power-Battery 
def pack_battery_conditions(conditions,battery,avionics_payload_power,P): 
    """ Packs the results from the network into propulsion data structures.
    
        Assumptions:
        None
    
        Source:
        N/A
    
        Inputs: 
        battery.
               inputs.current                  [Amperes]
               current_energy                  [Joules]
               voltage_open_circuit            [Volts]
               voltage_under_load              [Volts] 
               inputs.power_in                 [Watts]
               max_energy                      [Joules]
               cell_charge_throughput          [Ampere-hours]
               age                             [days]
               internal_resistance             [Ohms]
               state_of_charge                 [unitless]
               pack_temperature                [Kelvin]
               mass_properties.mass            [kilograms] 
               cell_voltage_under_load         [Volts]  
               cell_voltage_open_circuit       [Volts]  
               cell_current                    [Amperes]
               cell_temperature                [Kelvin] 
               heat_energy_generated           [Joules]
               cell_joule_heat_fraction        [unitless]  
               cell_entropy_heat_fraction      [unitless] 
               
        Outputs:
            conditions.propulsion.
               battery_current                    [Amperes]
               battery_energy                     [Joules]     
               battery_voltage_open_circuit       [Volts]     
               battery_voltage_under_load         [Volts]      
               battery_power_draw                 [Watts]     
               battery_max_aged_energy            [Joules]        
               battery_cycle_day                  [unitless]
               battery_internal_resistance        [Ohms]
               battery_state_of_charge            [unitless]
               battery_pack_temperature           [Kelvin]
               battery_efficiency                 [unitless]
               payload_efficiency                 [unitless]
               battery_specfic_power              [Watt-hours/kilogram]     
               electronics_efficiency             [unitless]
               battery_cell_power_draw            [Watts]
               battery_cell_energy                [Joules]
               battery_cell_voltage_under_load    [Volts]  
               battery_cell_voltage_open_circuit  [Volts]  
               battery_cell_current               [Amperes]
               battery_cell_temperature           [Kelvin]
               battery_cell_charge_throughput     [Ampere-Hours]
               battery_cell_heat_energy_generated [Joules]
               battery_cell_joule_heat_fraction   [unitless]  
               battery_cell_entropy_heat_fraction [unitless] 
    
        Properties Used:
        None
    """      
    n_series           = battery.pack_config.series  
    n_parallel         = battery.pack_config.parallel
    n_total            = n_series*n_parallel 
    battery_power_draw = battery.inputs.power_in    
    
    conditions.propulsion.battery_current                      = battery.inputs.current
    conditions.propulsion.battery_energy                       = battery.current_energy
    conditions.propulsion.battery_voltage_open_circuit         = battery.voltage_open_circuit
    conditions.propulsion.battery_voltage_under_load           = battery.voltage_under_load 
    conditions.propulsion.battery_power_draw                   = battery_power_draw 
    conditions.propulsion.battery_max_aged_energy              = battery.max_energy  
    conditions.propulsion.battery_cycle_day                    = battery.age
    conditions.propulsion.battery_internal_resistance          = battery.internal_resistance
    conditions.propulsion.battery_state_of_charge              = battery.state_of_charge 
    conditions.propulsion.battery_pack_temperature             = battery.pack_temperature  
    conditions.propulsion.battery_efficiency                   = (battery_power_draw+battery.resistive_losses)/battery_power_draw
    conditions.propulsion.payload_efficiency                   = (battery_power_draw+avionics_payload_power)/battery_power_draw            
    conditions.propulsion.battery_specfic_power                = -battery_power_draw/battery.mass_properties.mass    
    conditions.propulsion.electronics_efficiency               = -(P)/battery_power_draw    
    conditions.propulsion.battery_cell_power_draw              = battery.inputs.power_in /n_series
    conditions.propulsion.battery_cell_energy                  = battery.current_energy/n_total   
    conditions.propulsion.battery_cell_voltage_under_load      = battery.cell_voltage_under_load    
    conditions.propulsion.battery_cell_voltage_open_circuit    = battery.cell_voltage_open_circuit  
    conditions.propulsion.battery_cell_current                 = abs(battery.cell_current)        
    conditions.propulsion.battery_cell_temperature             = battery.cell_temperature
    conditions.propulsion.battery_cell_charge_throughput       = battery.cell_charge_throughput
    conditions.propulsion.battery_cell_heat_energy_generated   = battery.heat_energy_generated
    conditions.propulsion.battery_cell_joule_heat_fraction     = battery.cell_joule_heat_fraction   
    conditions.propulsion.battery_cell_entropy_heat_fraction   = battery.cell_entropy_heat_fraction 
    
    return 