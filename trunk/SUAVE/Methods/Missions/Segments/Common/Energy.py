## @ingroup Methods-Missions-Segments-Common
# Energy.py
# 
# Created:  Jul 2014, SUAVE Team
# Modified: Jan 2016, E. Botero
#           Jul 2017, E. Botero
#           Aug 2021, M. Clarke
#           Oct 2021, E. Botero

# ----------------------------------------------------------------------
#  Initialize Battery
# ----------------------------------------------------------------------

## @ingroup Methods-Missions-Segments-Common
def initialize_battery(segment):
    """ Sets the initial battery energy at the start of the mission
    
        Assumptions:
        N/A
        
        Inputs:
            segment.state.initials.conditions:
                propulsion.battery_energy    [Joules]
            segment.battery_energy           [Joules]
            
        Outputs:
            segment.state.conditions:
                propulsion.battery_energy    [Joules]

        Properties Used:
        N/A
                                
    """ 
       
    if segment.state.initials:
        initial_mission_energy               = segment.state.initials.conditions.propulsion.battery_max_initial_energy
        battery_max_aged_energy              = segment.state.initials.conditions.propulsion.battery_max_aged_energy         
        initial_segment_energy               = segment.state.initials.conditions.propulsion.battery_energy[-1,0]
        initial_pack_temperature             = segment.state.initials.conditions.propulsion.battery_pack_temperature[-1,0]
        battery_cell_charge_throughput       = segment.state.initials.conditions.propulsion.battery_cell_charge_throughput[-1,0]  
        battery_cycle_day                    = segment.state.initials.conditions.propulsion.battery_cycle_day        
        battery_discharge_flag               = segment.battery_discharge 
        battery_resistance_growth_factor     = segment.state.initials.conditions.propulsion.battery_resistance_growth_factor
        battery_thevenin_voltage             = segment.state.initials.conditions.propulsion.battery_thevenin_voltage[-1,0]  
        battery_capacity_fade_factor         = segment.state.initials.conditions.propulsion.battery_capacity_fade_factor
        
        if battery_discharge_flag == False: 
            battery_max_aged_energy  = initial_mission_energy*battery_capacity_fade_factor    
        
        segment.state.conditions.propulsion.battery_max_initial_energy                 = initial_mission_energy
        segment.state.conditions.propulsion.battery_energy[:,0]                        = initial_segment_energy 
        segment.state.conditions.propulsion.battery_max_aged_energy                    = battery_max_aged_energy    
        segment.state.conditions.propulsion.battery_pack_temperature[:,0]              = initial_pack_temperature
        segment.state.conditions.propulsion.battery_cycle_day                          = battery_cycle_day        
        segment.state.conditions.propulsion.battery_cell_charge_throughput[:,0]        = battery_cell_charge_throughput 
        segment.state.conditions.propulsion.battery_discharge_flag                     = battery_discharge_flag
        segment.state.conditions.propulsion.battery_resistance_growth_factor           = battery_resistance_growth_factor 
        segment.state.conditions.propulsion.battery_thevenin_voltage[:,0]              = battery_thevenin_voltage 
        segment.state.conditions.propulsion.battery_capacity_fade_factor               = battery_capacity_fade_factor      
    

# ----------------------------------------------------------------------
#  Update Thrust
# ----------------------------------------------------------------------

## @ingroup Methods-Missions-Segments-Common
def update_thrust(segment):
    """ Evaluates the energy network to find the thrust force and mass rate

        Inputs -
            segment.analyses.energy_network    [Function]

        Outputs -
            state.conditions:
               frames.body.thrust_force_vector [Newtons]
               weights.vehicle_mass_rate       [kg/s]


        Assumptions -


    """    
    
    # unpack
    energy_model = segment.analyses.energy

    # evaluate
    results   = energy_model.evaluate_thrust(segment.state)

    # pack conditions
    conditions = segment.state.conditions
    conditions.frames.body.thrust_force_vector = results.thrust_force_vector
    conditions.weights.vehicle_mass_rate       = results.vehicle_mass_rate
    

## @ingroup Methods-Missions-Segments-Common
def update_battery(segment):
    """ This just runs the battery module. All results are implicit to the battery model

        Inputs -
            segment.analyses.energy_network    [Function]

        Outputs -


        Assumptions -


    """    
    
    # unpack
    energy_model = segment.analyses.energy

    # evaluate
    energy_model.evaluate_thrust(segment.state)
    

def update_battery_state_of_health(segment):  
    """Updates battery age based on operating conditions, cell temperature and time of operation.
       Source: 
       Cell specific. See individual battery cell for more details
         
       Assumptions:
       Cell specific. See individual battery cell for more details
      
       Inputs: 
       segment.
           conditions                    - conditions of battery at each segment  [unitless]
           increment_battery_cycle_day   - flag to increment battery cycle day    [boolean]
       
       Outputs:
       N/A  
            
       Properties Used:
       N/A 
    """ 
    increment_day = segment.increment_battery_cycle_day
    
    for network in segment.analyses.energy.network: 
        battery = network.battery
        battery.update_battery_state_of_health(segment,increment_battery_cycle_day = increment_day) 