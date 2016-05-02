# size_weights_given_mission_range.py
#
# Created:  Sep 2014, T. Orra and C. Ilario
# Modified: Jan 2016, E. Botero
# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------

from SUAVE.Methods.Performance.size_mission_range_given_weights import size_mission_range_given_weights

import numpy as np

# ----------------------------------------------------------------------
#  Calculate vehicle Payload Range Diagram
# ----------------------------------------------------------------------

def size_weights_given_mission_range(vehicle,mission,cruise_segment_tag,mission_payload,target_range,reserve_fuel=0.):
    """ SUAVE.Methods.Performance.size_weights_given_mission_range(vehicle,mission,cruise_segment_tag,mission_payload,takeoff_weight,reserve_fuel=0.):
        Calculates vehicle weight for a given range with given payload

        Inputs:
            vehicle                     - SUave type vehicle
            mission                     - SUave type mission profile
            cruise_segment_tag          - Mission segment to be considered Cruise
            mission_payload             - float or 1d array with mission payload weight for each mission [kg]
            target_range                - float or 1d array with target range for each mission [m]
            reserve_fuel   [optional]   - float of 1d array with required reserve fuel [kg]

        Outputs:
            distance              - float or 1d array with Range results for each mission
            fuel                  - float or 1d array with fuel burn results for each mission
			tow					  - float or 1d array with takeoff weight for each mission

        Assumptions:
            Constant altitude cruise.

    """
    #unpack
    masses = vehicle.mass_properties

    OEW = masses.operating_empty
    if not OEW:
        print "Error calculating fuel for given mission: Vehicle Operating Empty not defined"
        return True

    MZFW = vehicle.mass_properties.max_zero_fuel
    if not MZFW:
        print "Error calculating fuel for given mission: Vehicle MZFW not defined"
        return True

    MaxPLD = vehicle.mass_properties.max_payload
    if not MaxPLD:
        MaxPLD = MZFW - OEW  # If payload max not defined, calculate based in design weights

    MaxFuel = vehicle.mass_properties.max_fuel
    if not MaxFuel:
        MaxFuel = vehicle.mass_properties.max_takeoff - OEW # If not defined, calculate based in design weights
        if MaxFuel < 0. :
            print "Error calculating fuel for given mission: Vehicle MTOW not defined"
            return True

    # Defining arrays for input and output
    mission_payload = np.atleast_1d(mission_payload)
    target_range    = np.atleast_1d(target_range)
    reserve_fuel    = np.atleast_1d(reserve_fuel)

    # Check if # payload input is equal to # target_range
    if len(mission_payload) == 1 and len(target_range) > 1:
        mission_payload = np.multiply(np.ones_like(target_range),mission_payload[0])

    # Check if # reserve_fuel input is equal to # takeoff weights
    if len(reserve_fuel) == 1 and len(target_range) > 1:
        reserve_fuel = np.multiply(np.ones_like(target_range),reserve_fuel[0])

    # Allocating variabels
    fuel     = np.zeros_like(mission_payload)
    distance = np.zeros_like(mission_payload)
    tow      = np.zeros_like(mission_payload)

    # Locate cruise segment to be variated
    for i in range(len(mission.segments)):          #loop for all segments
        if mission.segments[i].tag.upper() == cruise_segment_tag.upper() :
            segmentNum = i
            break

    for id,range_id in enumerate(target_range):
        payload = mission_payload[id]
        reserve  = reserve_fuel[id]

        residual = vehicle.mass_properties.max_takeoff
        tol = 5. # kg
        takeoff_weight = 0.
        iter = 0

        while abs(residual) > tol and iter < 10:

            iter = iter + 1
            takeoff_weight = takeoff_weight + residual
            dist_id,fuel_id = size_mission_range_given_weights(vehicle,mission,cruise_segment_tag,payload,takeoff_weight,reserve)
            residual = (range_id - dist_id) * float(fuel_id)/float(dist_id)

        distance[id] = dist_id
        fuel[id]     = fuel_id
        tow[id]      = takeoff_weight

    return distance,fuel,tow