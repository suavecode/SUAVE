
# ----------------------------------------------------------------------
#   Imports
# ----------------------------------------------------------------------

import SUAVE
from SUAVE.Attributes import Units

import numpy as np
import pylab as plt

import copy, time

from SUAVE.Structure import (
Data, Container, Data_Exception, Data_Warning,
)

# ----------------------------------------------------------------------
#   Optimization Setup
# ----------------------------------------------------------------------

def setup_interface(configs,analyses):
    
    # instantiate interface
    
    interface = SUAVE.Optimization.Interface()
    interface.configs      = configs
    interface.analyses     = analyses
    
    strategy = interface.strategy
    strategy.unpack_inputs = unpack_inputs
    strategy.finalize      = full_finalize
    
    missions = missions_setup()
    strategy.missions      = missions
    
    strategy.field_length  = field_length
    strategy.noise         = noise
    strategy.performance   = SUAVE.Methods.Performance.evaluate_performance
    strategy.summarize     = summarize
    
    return interface
    
    
def missions_setup(base_mission):
    
    missions = SUAVE.Analyses.Missions.Mission.Container()
    missions.base = base_mission
    
    fuel_mission = SUAVE.Analyses.Missions.Fuel_Constrained()
    fuel_mission.tag = 'fuel'
    fuel_mission.mission = base_mission
    missions.append(fuel_mission)
    
    short_field = SUAVE.Analyses.Missions.Short_Field_Constrained()
    short_field.tag = 'short_field'
    short_field.mission = base_mission
    missions.append(short_field)

    payload = SUAVE.Analyses.Missions.Payload_Constrained()
    payload.tag = 'payload'
    payload.mission = base_mission
    missions.append(payload)
    
    return missions    
    
    
    
def unpack_inputs(interface,inputs):
    
    vehicle = interface.configs.base
    vehicle.wings.main_wing.spans.project   = inputs.projected_span
    vehicle.wings.fuselages.fuselage.length = inputs.fuselage_length
    
    mission = interface.missions.base
    mission.segments.cruise.distance        = inputs.cruise_distance
    

def full_finalize(interface):
    
    configs = interface.configs
    configs.finalize()
    
    analyses = interface.analyses
    analyses.finalize()
    
    strategy = interface.strategy
    strategy.finalize()
    
    
    
def field_length(interface):
    
    estimate_tofl = SUAVE.Methods.Performance.estimate_take_off_field_length
    
    configs  = interface.configs
    missions = interface.strategy.missions
    
    takeoff_airport = missions.base.segments.takeoff.airport
    
    results = estimate_tofl( configs.landing , takeoff_airport )
    
    return results

def noise(interface):
    
    evaluate_noise = SUAVE.Methods.Noise.Correlations.shevell
    
    vehicle = interface.configs.base
    results = interface.results
    mission_profile = results.missions.base.profile
    
    weight_landing    = mission_profile.segments[-1].conditions.weights.total_mass[-1,0]
    number_of_engines = vehicle.propulsors['Turbo Fan'].number_of_engines
    thrust_sea_level  = vehicle.propulsors['Turbo Fan'].design_thrust
    thrust_landing    = mission_profile.segments[-1].conditions.frames.body.thrust_force_vector[-1,0]
    
    # evaluate
    results = evaluate_noise( weight_landing    , 
                            number_of_engines , 
                            thrust_sea_level  , 
                            thrust_landing     )
    
    return results
    
    
def summarize(interface):
    
    results = interface.results
    
    summary = SUAVE.Attributes.Results()
    
    summary.this_value = results.hello_world
    
    return summary