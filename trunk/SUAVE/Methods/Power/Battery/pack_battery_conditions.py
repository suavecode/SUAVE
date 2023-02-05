## @ingroup Methods-Power-Battery 
# pack_battery_conditions.py
# 
# Created: Sep 2021, M. Clarke 

# ----------------------------------------------------------------------
#  Methods
# ----------------------------------------------------------------------
## @ingroup Methods-Power-Battery 
def pack_battery_conditions(conditions,battery,avionics_payload_power): 
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
               age                             [days]
               internal_resistance             [Ohms]
               state_of_charge                 [unitless]
               pack.temperature                [Kelvin]
               mass_properties.mass            [kilograms] 
               heat_energy_generated           [Joules]
               cell.charge_throughput          [Ampere-hours]
               cell.voltage_under_load         [Volts]  
               cell.voltage_open_circuit       [Volts]  
               cell.current                    [Amperes]
               cell.temperature                [Kelvin] 
               cell.joule_heat_fraction        [unitless]  
               cell.entropy_heat_fraction      [unitless] 
               
        Outputs:
            conditions.propulsion.   
               electronics_efficiency             [unitless]
               payload_efficiency                 [unitless]
               battery.pack.specfic_power         [Watt-hours/kilogram]  
                       current                    [Amperes]
                       energy                     [Joules]     
                       voltage_open_circuit       [Volts]     
                       voltage_under_load         [Volts]      
                       power_draw                 [Watts]     
                       max_aged_energy            [Joules]        
                       cycle_day                  [unitless]
                       internal_resistance        [Ohms]
                       state_of_charge            [unitless]
                       pack.temperature           [Kelvin]
                       efficiency                 [unitless]
                       pack.heat_energy_generated [Joules]
                       cell.power                 [Watts]
                       cell.energy                [Joules]
                       cell.voltage_under_load    [Volts]  
                       cell.voltage_open_circuit  [Volts]  
                       cell.current               [Amperes]
                       cell.temperature           [Kelvin]
                       cell.charge_throughput     [Ampere-Hours]
                       cell.joule_heat_fraction   [unitless]  
                       cell.entropy_heat_fraction [unitless] 
    
        Properties Used:
        None
    """      
    n_series           = battery.pack.electrical_configuration.series  
    n_parallel         = battery.pack.electrical_configuration.parallel
    n_total            = n_series*n_parallel 
    battery_power_draw = battery.inputs.power_in    
       
    conditions.propulsion.payload_efficiency                   = (battery_power_draw+avionics_payload_power)/battery_power_draw  
    conditions.propulsion.battery.pack.current                 = battery.inputs.current
    conditions.propulsion.battery.pack.energy                  = battery.pack.current_energy
    conditions.propulsion.battery.pack.voltage_open_circuit    = battery.pack.voltage_open_circuit
    conditions.propulsion.battery.pack.voltage_under_load      = battery.pack.voltage_under_load 
    conditions.propulsion.battery.pack.power_draw              = battery_power_draw 
    conditions.propulsion.battery.pack.max_aged_energy         = battery.pack.max_energy  
    conditions.propulsion.battery.pack.temperature             = battery.pack.temperature  
    conditions.propulsion.battery.pack.heat_energy_generated   = battery.pack.heat_energy_generated
    conditions.propulsion.battery.pack.efficiency              = (battery_power_draw+battery.pack.resistive_losses)/battery_power_draw          
    conditions.propulsion.battery.pack.specfic_power           = -battery_power_draw/battery.mass_properties.mass   
    conditions.propulsion.battery.pack.internal_resistance     = battery.pack.internal_resistance 
    conditions.propulsion.battery.cell.cycle_in_day            = battery.cell.age
    conditions.propulsion.battery.cell.state_of_charge         = battery.cell.state_of_charge 
    conditions.propulsion.battery.cell.power                   = battery.inputs.power_in/n_series
    conditions.propulsion.battery.cell.energy                  = battery.pack.current_energy/n_total   
    conditions.propulsion.battery.cell.voltage_under_load      = battery.cell.voltage_under_load    
    conditions.propulsion.battery.cell.voltage_open_circuit    = battery.cell.voltage_open_circuit  
    conditions.propulsion.battery.cell.current                 = abs(battery.cell.current)        
    conditions.propulsion.battery.cell.temperature             = battery.cell.temperature
    conditions.propulsion.battery.cell.charge_throughput       = battery.cell.charge_throughput
    conditions.propulsion.battery.cell.joule_heat_fraction     = battery.cell.joule_heat_fraction   
    conditions.propulsion.battery.cell.entropy_heat_fraction   = battery.cell.entropy_heat_fraction 
    
    return 