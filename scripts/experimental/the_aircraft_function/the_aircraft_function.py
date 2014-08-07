# tut_mission_Boeing_737.py
# 
# Created:  Michael Colonno, Apr 2013
# Modified: Michael Vegh   , Sep 2013
#           Trent Lukaczyk , Jan 2014

""" The main evaluation function
"""


# ----------------------------------------------------------------------
#   Imports
# ----------------------------------------------------------------------

import SUAVE
from SUAVE.Attributes import Units
from SUAVE.Structure import Data
from compile_results import compile_results

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
    vehicle.Mass_Props.breakdown = breakdown
    vehicle.Mass_Props.m_empty = vehicle.Mass_Props.breakdown.empty
    
    for config in vehicle.Configs:
        config.Mass_Props.m_empty = vehicle.Mass_Props.breakdown.empty
    
    results.weight_breakdown = breakdown
    
    return results


# ----------------------------------------------------------------------
#   Evaluate the Field Length
# ----------------------------------------------------------------------

def evaluate_field_length(vehicle,mission,results):
    
    # unpack
    airport = mission.airport
    
    takeoff_config = vehicle.Configs.takeoff
    landing_config = vehicle.Configs.landing
    
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
    mission_profile = results.mission_profile
    
    weight_landing    = mission_profile.Segments[-1].conditions.weights.total_mass[-1,0]
    number_of_engines = vehicle.Propulsors['Turbo Fan'].no_of_engines
    thrust_sea_level  = vehicle.Propulsors['Turbo Fan'].design_thrust
    thrust_landing    = mission_profile.Segments[-1].conditions.frames.body.thrust_force_vector[-1,0]
    
    from SUAVE.Methods.Noise.Correlations import shevell as evaluate_noise
    
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

#def compile_results(vehicle,mission,results):
    
    # unpack
    
    # evaluate
    
    # pack
    
    #return results


# ---------------------------------------------------------------------- 
#   Call Main
# ----------------------------------------------------------------------

if __name__ == '__main__':

    main()
    
    
  