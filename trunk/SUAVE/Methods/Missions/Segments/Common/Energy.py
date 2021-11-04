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
       
    conditions = segment.state.conditions.propulsion
    if segment.state.initials:

        initials   = segment.state.initials.conditions.propulsion
        
        initial_mission_energy       = initials.battery_max_initial_energy
        battery_max_aged_energy      = initials.battery_max_aged_energy         
        battery_discharge_flag       = segment.battery_discharge 
        battery_capacity_fade_factor = initials.battery_capacity_fade_factor
        
        if battery_discharge_flag == False: 
            battery_max_aged_energy  = initial_mission_energy*battery_capacity_fade_factor    
        
        conditions.battery_max_initial_energy          = initial_mission_energy
        conditions.battery_energy[:,0]                 = initials.battery_energy[-1,0]
        conditions.battery_max_aged_energy             = battery_max_aged_energy
        conditions.battery_pack_temperature[:,0]       = initials.battery_pack_temperature[-1,0]
        conditions.battery_cell_temperature[:,0]       = initials.battery_cell_temperature[-1,0]
        conditions.battery_cycle_day                   = initials.battery_cycle_day      
        conditions.battery_cell_charge_throughput[:,0] = initials.battery_cell_charge_throughput[-1,0]
        conditions.battery_discharge_flag              = battery_discharge_flag
        conditions.battery_resistance_growth_factor    = initials.battery_resistance_growth_factor
        conditions.battery_capacity_fade_factor        = battery_capacity_fade_factor 
    
    if 'battery_pack_temperature' in segment: # rewrite initial temperature of the battery if it is known 
        conditions.battery_pack_temperature[:,0]       = segment.battery_pack_temperature
        conditions.battery_cell_temperature[:,0]       = segment.battery_pack_temperature 
        
            
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