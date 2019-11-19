## @ingroup Methods-Missions-Segments-Common
# Energy.py
# 
# Created:  Jul 2014, SUAVE Team
# Modified: Jan 2016, E. Botero
#           Jul 2017, E. Botero

# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------

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
    
    
    if segment.state.initials:
        energy_initial            = segment.state.initials.conditions.propulsion.battery_energy[-1,0]
        temperature_initial       = segment.state.initials.conditions.propulsion.battery_temperature[-1,0]
        battery_charge_throughput = segment.state.initials.conditions.propulsion.battery_charge_throughput
        battery_age_in_days       = segment.state.initials.conditions.propulsion.battery_age_in_days
    elif 'battery_energy' in segment:
        energy_initial            = segment.battery_energy
        temperature_initial       = segment.battery_temperature
        battery_age_in_days       = segment.battery_age_in_days
        battery_charge_throughput = segment.battery_charge_throughput
    else:
        energy_initial            = 0.0
        temperature_initial       = 0.0
        battery_age_in_days       = 1
        battery_charge_throughput = 0.0
        
    segment.state.conditions.propulsion.battery_energy[:,0]       = energy_initial
    segment.state.conditions.propulsion.battery_temperature[:,0]  = temperature_initial
    segment.state.conditions.propulsion.battery_age_in_days       = battery_age_in_days
    segment.state.conditions.propulsion.battery_charge_throughput = battery_charge_throughput 
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