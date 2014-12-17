
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
#   Main
# ----------------------------------------------------------------------

def main():
    
    # setup the interface
    interface = setup_interface()
    
    inputs = Data()
    inputs.projected_span  = 20.
    inputs.fuselage_length = 40.
    inputs.cruise_distance = 1000.
    
    # evalute!
    interface.evaluate(inputs)


# ----------------------------------------------------------------------
#   Optimization Interface Setup
# ----------------------------------------------------------------------

def setup_interface():
    
    # ------------------------------------------------------------------
    #   Instantiate Interface
    # ------------------------------------------------------------------
    
    interface = SUAVE.Optimization.Interface()
    
    # ------------------------------------------------------------------
    #   Vehicle and Analyses Information
    # ------------------------------------------------------------------
    
    from full_setup import full_setup
    
    configs,analyses = full_setup()
    
    interface.configs  = configs
    interface.analyses = analyses
    
    
    # ------------------------------------------------------------------
    #   Analysis Process
    # ------------------------------------------------------------------
    
    process = interface.process
    
    # the input unpacker
    process.unpack_inputs = unpack_inputs
    
    # size the base config
    process.simple_sizing = simple_sizing
    
    # finalizes the data dependencies
    process.finalize = finalize
    
    # the missions
    process.missions = missions
    
    # varius performance studies
    process.field_length = field_length
    process.noise = noise
    process.performance  = SUAVE.Methods.Performance.evaluate_mission
    
    # summarize the results
    process.summary = summarize
    
    # done!
    return interface
    
    
# ----------------------------------------------------------------------
#   Unpack Inputs Step
# ----------------------------------------------------------------------
    
def unpack_inputs(interface):
    
    inputs = interface.inputs
    
    # apply the inputs
    vehicle = interface.configs.base
    vehicle.wings['Main Wing'].spans.project = inputs.projected_span
    vehicle.fuselages.Fuselage.length        = inputs.fuselage_length
    
    mission = interface.strategy.missions.base
    mission.segments.cruise.distance = inputs.cruise_distance


# ----------------------------------------------------------------------
#   Apply Simple Sizing Principles
# ----------------------------------------------------------------------

def simple_sizing(interface):
    
    # ------------------------------------------------------------------
    #   Base Vehicle Configuration
    # ------------------------------------------------------------------
    vehicle = interface.configs.base
    
    # zero fuel weight
    vehicle.mass_properties.max_zero_fuel = 0.9 * vehicle.mass_properties.max_takeoff 
    
    # wing areas
    for wing in vehicle.wings:
        wing.areas.wetted   = 2.0 * wing.areas.reference
        wing.areas.exposed  = 0.8 * wing.areas.wetted
        wing.areas.affected = 0.6 * wing.areas.wetted
    
    # fuselage seats
    vehicle.fuselages.fuselage.number_coach_seats = vehicle.passengers
    
    # ------------------------------------------------------------------
    #   Landing Configuration
    # ------------------------------------------------------------------
    landing = interface.configs.landing
    
    # landing weight
    landing.mass_properties.landing = 0.85 * vehicle.mass_properties.takeoff
    
    # done!
    return interface


# ----------------------------------------------------------------------
#   Finalizing Function (make part of optimization interface)[needs to come after simple sizing doh]
# ----------------------------------------------------------------------    

def finalize(interface):
    
    interface.configs.finalize()
    interface.analyses.finalize()
    interface.process.finalize()


# ----------------------------------------------------------------------
#   Process Missions
# ----------------------------------------------------------------------    

def missions(interface):
    
    missions = interface.process.missions
    
    results = missions.evaluate()
    
    return results
            
    
# ----------------------------------------------------------------------
#   Field Length Evaluation
# ----------------------------------------------------------------------    
    
def field_length(interface):
    
    # unpack tofl analysis
    estimate_tofl = SUAVE.Methods.Performance.estimate_take_off_field_length
    
    # unpack data
    configs  = interface.configs
    missions = interface.strategy.missions
    takeoff_airport = missions.base.segments.takeoff.airport
    
    # evaluate
    results = estimate_tofl( configs.landing , takeoff_airport )
    
    return results


# ----------------------------------------------------------------------
#   Noise Evaluation
# ----------------------------------------------------------------------    

def noise(interface):
    
    # TODO - use the analysis
    
    # unpack noise analysis
    evaluate_noise = SUAVE.Methods.Noise.Correlations.shevell
    
    # unpack data
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
    
    
# ----------------------------------------------------------------------
#   Summarize the Data
# ----------------------------------------------------------------------    

def summarize(interface):
    
    results = interface.results
    
    summary = SUAVE.Attributes.Results()
    
    summary.this_value = results.hello_world
    
    return summary




if __name__ == '__main__':
    main()
    
    
    
    
    ## ------------------------------------------------------------------
    ##   Tests
    ## ------------------------------------------------------------------    
    
    #inputs = Data()
    #inputs.projected_span  = 1.0
    #inputs.fuselage_length = 1.0
    #inputs.cruise_distance = 1.0
    
    #strategy.unpack_inputs(interface,inputs)
    
    #import cPickle as pickle
    #i = pickle.dumps(strategy)
    #p = pickle.loads(i)
    
