# the_aircraft_function.py
# 
# Created:  Michael Colonno, Apr 2013
# Modified: Michael Vegh   , Sep 2013
#           Trent Lukaczyk , Jan 2014
#           Tim MacDonald  , Sep 2014

""" The main evaluation function
"""


# ----------------------------------------------------------------------
#   Imports
# ----------------------------------------------------------------------

import SUAVE
from SUAVE.Core import Units
from SUAVE.Core import Data

import numpy as np

import copy, time


# ----------------------------------------------------------------------
#   The Aircraft Function
# ----------------------------------------------------------------------

def the_aircraft_function(vehicle,mission):
    
    # start the results data structure
    results = Data()
    
    # evaluate weights
    results = evaluate_weights(vehicle,results)
    
    print results

    # evaluate field length
    results = evaluate_field_length(vehicle,mission,results)
    
    # evaluate the mission
    results = evaluate_mission(vehicle,mission,results)

    # evaluate noise
    results = evaluate_noise(vehicle,mission,results)
    
    # compile results
    results = compile_results(vehicle,mission,results)
    
    return results


# ----------------------------------------------------------------------
#   Evaluate Aircraft Weights
# ----------------------------------------------------------------------

def evaluate_weights(vehicle,results):
    
    # unpack 
    from SUAVE.Methods.Weights.Correlations.Tube_Wing import empty
    
    # evaluate
    breakdown = empty(vehicle)
     
    # pack
    vehicle.mass_properties.breakdown = breakdown
    vehicle.mass_properties.operating_empty = vehicle.mass_properties.breakdown.empty
    
    for config in vehicle.configs:
        config.mass_properties.operating_empty = vehicle.mass_properties.breakdown.empty
    
    results.weight_breakdown = breakdown
    
    return results


# ----------------------------------------------------------------------
#   Evaluate the Field Length
# ----------------------------------------------------------------------

def evaluate_field_length(vehicle,mission,results):
    
    # unpack
    airport = mission.airport
    
    takeoff_config = vehicle.configs.takeoff
    landing_config = vehicle.configs.landing
    
    from SUAVE.Methods.Performance import estimate_take_off_field_length
    from SUAVE.Methods.Performance import estimate_landing_field_length    
    
    # evaluate
    TOFL = estimate_take_off_field_length(vehicle,takeoff_config,airport)
    LFL = estimate_landing_field_length(vehicle,landing_config,airport)
    
    # pack
    field_length = Data()
    field_length.takeoff = TOFL[0]
    field_length.landing = LFL[0]
    
    results.field_length = field_length
    
    return results


# ----------------------------------------------------------------------
#   Evaluate the Mission
# ----------------------------------------------------------------------

def evaluate_mission(vehicle,mission,results):
    
    # unpack
    from SUAVE.Methods.Performance import evaluate_mission
    
    # evaluate
    mission_profile = evaluate_mission(mission)
    
    # pack
    results.mission_profile = mission_profile
    
    return results


# ----------------------------------------------------------------------
#   Evaluate Noise Emissions
# ----------------------------------------------------------------------

def evaluate_noise(vehicle,mission,results):
    
    # unpack
    from SUAVE.Methods.Noise.Correlations import shevell as evaluate_noise
    
    mission_profile = results.mission_profile
    
    weight_landing    = mission_profile.segments[-1].conditions.weights.total_mass[-1,0]
    number_of_engines = vehicle.propulsors['turbo_fan'].number_of_engines
    thrust_sea_level  = vehicle.propulsors['turbo_fan'].thrust.design
    thrust_landing    = mission_profile.segments[-1].conditions.frames.body.thrust_force_vector[-1,0]
    
    # evaluate
    noise = evaluate_noise( weight_landing    , 
                            number_of_engines , 
                            thrust_sea_level  , 
                            thrust_landing     )
    
    # pack
    results.noise = noise
    
    return results


# ----------------------------------------------------------------------
#   Compile Useful Results
# ----------------------------------------------------------------------

def compile_results(vehicle,mission,results):
    
    # merge all segment conditions
    def stack_condition(a,b):
        if isinstance(a,np.ndarray):
            return np.vstack([a,b])
        else:
            return None

    conditions = None
    for segment in results.mission_profile.segments:
        if conditions is None:
            conditions = segment.conditions
            continue
        conditions = conditions.do_recursive(stack_condition,segment.conditions)
      
    # pack
    results.output = Data()
    results.output.stability = Data()
    results.output.weight_empty = vehicle.mass_properties.operating_empty
    results.output.fuel_burn = max(conditions.weights.total_mass[:,0]) - min(conditions.weights.total_mass[:,0])
    #results.output.max_usable_fuel = vehicle.mass_properties.max_usable_fuel
    results.output.noise = results.noise    
    results.output.mission_time_min = max(conditions.frames.inertial.time[:,0] / Units.min)
    results.output.max_altitude_km = max(conditions.freestream.altitude[:,0] / Units.km)
    results.output.range_nmi = results.mission_profile.segments[-1].conditions.frames.inertial.position_vector[-1,0] / Units.nmi
    results.output.field_length = results.field_length
    results.output.stability.cm_alpha = max(conditions.aerodynamics.cm_alpha[:,0])
    results.output.stability.cn_beta = max(conditions.aerodynamics.cn_beta[:,0])

    
    #TODO: revisit how this is calculated
    results.output.second_segment_climb_rate = results.mission_profile.segments['Climb - 2'].climb_rate

    
    return results


# ---------------------------------------------------------------------- 
#   Call Main
# ----------------------------------------------------------------------

if __name__ == '__main__':

    main()
    
    
  
