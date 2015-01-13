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
from SUAVE.Core import Units
from SUAVE.Core import Data

import numpy as np

import copy, time


# ----------------------------------------------------------------------
#   The Aircraft Function
# ----------------------------------------------------------------------

def the_aircraft_function_EMBRAER(vehicle,mission):

    # start the results data structure
    results = Data()

    # evaluate weights
    print ' Estimating weights'
    results = evaluate_weights(vehicle,results)

    # evaluate field length
    print ' Evaluate field length'
    results = evaluate_field_length(vehicle,mission,results)

    # evaluate range for required payload
    print ' Evaluate range for required payload'
    results = evaluate_range_for_design_payload(vehicle,mission,results)

    # evaluate range from short field
    print ' Evaluate range from short field'
    results = evaluate_range_from_short_field(vehicle,mission,results)

    # evaluate design mission for fuel consuption
    print ' Evaluate design mission for fuel consuption'
    results = evaluate_mission_for_fuel(vehicle,mission,results)

    # evaluate noise
    print ' Evaluate noise'
    results = evaluate_noise(vehicle,mission,results)

    # compile results
    results = compile_results(vehicle,mission,results)

    print results.field_length
    print results.design_payload_mission
    print results.short_field
    print results.mission_for_fuel


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
    field_length.tag = 'field_length'
    field_length.takeoff = TOFL[0]
    field_length.landing = LFL[0]

    results.field_length = field_length

    return results

# ----------------------------------------------------------------------
#   Evaluate Ranges
# ----------------------------------------------------------------------

def evaluate_range_for_design_payload(vehicle,mission,results):

    # SUave import
    from SUAVE.Methods.Performance import size_mission_range_given_weights

    # unpack
    cruise_segment_tag = 'Cruise'
    mission_payload = vehicle.mass_properties.payload
    takeoff_weight  = vehicle.mass_properties.takeoff

    # Call function
    distance,fuel = size_mission_range_given_weights(vehicle,mission,cruise_segment_tag,mission_payload,takeoff_weight)

    results.design_payload_mission = Data()
    results.design_payload_mission.tag = 'design_payload_mission'
    results.design_payload_mission.payload = mission_payload
    results.design_payload_mission.range   = distance
    results.design_payload_mission.fuel    = fuel

    return results

# ----------------------------------------------------------------------
#   Evaluate Range from short field
# ----------------------------------------------------------------------
def evaluate_range_from_short_field(vehicle,mission,results):

    # unpack
    airport_short_field = mission.airport_short_field
    tofl = airport_short_field.field_lenght
    takeoff_config = vehicle.configs.takeoff

    from SUAVE.Methods.Performance import find_takeoff_weight_given_tofl

    # evaluate maximum allowable takeoff weight from a short field
    tow_short_field = find_takeoff_weight_given_tofl(vehicle,takeoff_config,airport_short_field,tofl)

    # determine maximum range based in tow short_field

    from SUAVE.Methods.Performance import size_mission_range_given_weights
    # unpack
    cruise_segment_tag = 'Cruise'
    mission_payload = vehicle.mass_properties.payload
    # call function
    distance,fuel = size_mission_range_given_weights(vehicle,mission,cruise_segment_tag,mission_payload,tow_short_field)

    # pack
    short_field = Data()
    short_field.tag            = 'short_field'
    short_field.takeoff_weight = tow_short_field
    short_field.range          = distance
    short_field.fuel           = fuel

    results.short_field = short_field

    return results

# ----------------------------------------------------------------------
#   run mission for fuel consumption evaluation
# ----------------------------------------------------------------------
def evaluate_mission_for_fuel(vehicle,mission,results):

    # unpack
    cruise_segment_tag = 'Cruise'
    mission_payload = vehicle.mass_properties.payload
    target_range = mission.range_for_fuel

    from SUAVE.Methods.Performance import size_weights_given_mission_range
    from SUAVE.Methods.Performance import evaluate_mission

    # call function
    distance,fuel,tow = size_weights_given_mission_range(vehicle,mission,cruise_segment_tag,mission_payload,target_range)

    mission_profile = evaluate_mission(mission)

    # pack
    results.mission_for_fuel                = Data()
    results.mission_for_fuel.tag            = 'mission_for_fuel'
    results.mission_for_fuel.takeoff_weight = tow
    results.mission_for_fuel.range          = distance
    results.mission_for_fuel.fuel           = fuel
    results.mission_profile                 = mission_profile

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
##    results.output.second_segment_climb_rate = results.mission_profile.segments['Climb - 2'].climb_rate


    return results


# ----------------------------------------------------------------------
#   Call Main
# ----------------------------------------------------------------------

if __name__ == '__main__':

    main()


