## @ingroup Methods-Missions-Segments-Common
# Energy.py
# 
# Created:  Jul 2014, SUAVE Team
# Modified: Jan 2016, E. Botero
#           Jul 2017, E. Botero

# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------
import SUAVE
import numpy as np

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
    if 'battery_discharge' not in segment:
        segment.battery_discharge = True 

    if 'battery_energy' in segment:
        if 'ambient_temperature' not in segment: 
            segment.ambient_temperature  = segment.conditions.freestream.temperature[0,0] # TO DO !!  include segment.temperature_deviation
            
        if 'battery_cell_temperature' not in segment:          
            segment.battery_cell_temperature = segment.conditions.freestream.temperature[0,0]
            
        if 'battery_pack_temperature' not in segment:     
            segment.battery_pack_temperature = segment.conditions.freestream.temperature[0,0]
            
        if 'battery_charge_throughput' not in segment:
            segment.battery_charge_throughput = 0
       
        if 'battery_resistance_growth_factor' not in segment:
            segment.battery_resistance_growth_factor     = 1	 
            
        if 'battery_capacity_fade_factor' not in segment: 
            segment.battery_capacity_fade_factor         = 1           
            
        intial_segment_energy                = segment.battery_energy
        battery_max_aged_energy              = segment.battery_energy # segment.max_energy          
        initial_mission_energy               = segment.battery_energy 
        pack_temperature                     = segment.battery_pack_temperature
        battery_age_in_days                  = 0
        battery_charge_throughput            = segment.battery_charge_throughput
        battery_discharge                    = segment.battery_discharge   
        ambient_temperature                  = segment.ambient_temperature
        battery_resistance_growth_factor     = segment.battery_resistance_growth_factor
        battery_initial_thevenin_voltage     = 0.0
        battery_capacity_fade_factor         = segment.battery_capacity_fade_factor     
    
    elif segment.state.initials:
        initial_mission_energy               = segment.state.initials.conditions.propulsion.battery_max_initial_energy
        battery_max_aged_energy              = segment.state.initials.conditions.propulsion.battery_max_aged_energy         
        intial_segment_energy                = segment.state.initials.conditions.propulsion.battery_energy[-1,0]
        pack_temperature                     = segment.state.initials.conditions.propulsion.battery_pack_temperature[-1,0]
        battery_charge_throughput            = segment.state.initials.conditions.propulsion.battery_charge_throughput[-1,0]  
        battery_age_in_days                  = segment.battery_age_in_days
        battery_discharge                    = segment.battery_discharge
        ambient_temperature                  = segment.state.initials.conditions.propulsion.ambient_temperature
        battery_resistance_growth_factor     = segment.state.initials.conditions.propulsion.battery_resistance_growth_factor
        battery_initial_thevenin_voltage     = segment.state.initials.conditions.propulsion.battery_thevenin_voltage[-1,0]  
        battery_capacity_fade_factor         = segment.state.initials.conditions.propulsion.battery_capacity_fade_factor
                  
    else:
        initial_mission_energy               = 0.0
        intial_segment_energy                = 0.0
        pack_temperature                     = 292.65
        ambient_temperature                  = 292.65
        battery_age_in_days                  = 1
        battery_charge_throughput            = 0.0 
        battery_resistance_growth_factor     = 1.0
        battery_capacity_fade_factor         = 1.0  
        battery_initial_thevenin_voltage     = 0
        battery_discharge                    = True  
        
    if segment.battery_discharge == False: 
        battery_max_aged_energy  = initial_mission_energy*battery_capacity_fade_factor    
    
    segment.state.conditions.propulsion.battery_max_initial_energy                 = initial_mission_energy
    segment.state.conditions.propulsion.battery_energy[:,0]                        = intial_segment_energy 
    segment.state.conditions.propulsion.battery_max_aged_energy                    = battery_max_aged_energy    
    segment.state.conditions.propulsion.battery_pack_temperature[:,0]              = pack_temperature
    segment.state.conditions.propulsion.battery_age_in_days                        = battery_age_in_days
    segment.state.conditions.propulsion.battery_charge_throughput[:,0]             = battery_charge_throughput 
    segment.state.conditions.propulsion.battery_discharge                          = battery_discharge
    segment.state.conditions.propulsion.ambient_temperature                        = ambient_temperature
    segment.state.conditions.propulsion.battery_resistance_growth_factor           = battery_resistance_growth_factor 
    segment.state.conditions.propulsion.battery_initial_thevenin_voltage           = battery_initial_thevenin_voltage 
    segment.state.conditions.propulsion.battery_capacity_fade_factor               = battery_capacity_fade_factor      
    return

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

def update_battery_age(segment):  
    """This is an aging model for 18650 lithium-nickel-manganese-cobalt-oxide batteries. 
       
       Source: Schmalstieg, Johannes, et al. "A holistic aging model for Li (NiMnCo) O2
       based 18650 lithium-ion batteries." Journal of Power Sources 257 (2014): 325-334.
        
       Inputs:
         segment.conditions.propulsion. 
            t (battery age in days)                                                [days]   
            battery_cell_temperature                                               [Degrees Celcius] 
            battery_voltage_open_circuit                                           [Volts] 
            battery_charge_throughput                                   [Amp-hrs] 
            battery_state_of_charge                                                [unitless] 
       
       Outputs:
          segment.conditions.propulsion.
            battery_capacity_fade_factor     (internal resistance growth factor)   [unitless]
            battery_resistance_growth_factor (capactance (energy) growth factor)   [unitless]  
            
    """
    
    
    for network in segment.analyses.energy.network: 
        n_series = network.battery.pack_config.series 
    
    SOC        = segment.conditions.propulsion.battery_state_of_charge
    V_ul       = segment.conditions.propulsion.battery_voltage_under_load/n_series
    t          = segment.conditions.propulsion.battery_age_in_days 
    Q_prior    = segment.conditions.propulsion.battery_charge_throughput[-1,0] 
    Temp       = np.mean(segment.conditions.propulsion.battery_cell_temperature) 
    
    # aging model  
    delta_DOD = abs(SOC[0][0] - SOC[-1][0])
    rms_V_ul  = np.sqrt(np.mean(V_ul**2)) 
    alpha_cap = (7.542*np.mean(V_ul) - 23.75) * 1E6 * np.exp(-6976/(Temp))  
    alpha_res = (5.270*np.mean(V_ul) - 16.32) * 1E5 * np.exp(-5986/(Temp))  
    beta_cap  = 7.348E-3 * (rms_V_ul - 3.667)**2 +  7.60E-4 + 4.081E-3*delta_DOD
    beta_res  = 2.153E-4 * (rms_V_ul - 3.725)**2 - 1.521E-5 + 2.798E-4*delta_DOD
    
    E_fade_factor   = 1 - alpha_cap*(t**0.75) - beta_cap*np.sqrt(Q_prior)   
    R_growth_factor = 1 + alpha_res*(t**0.75) + beta_res*Q_prior 
    
    segment.conditions.propulsion.battery_capacity_fade_factor     = E_fade_factor  
    segment.conditions.propulsion.battery_resistance_growth_factor = R_growth_factor
    