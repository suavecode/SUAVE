# size_mission_range_given_weights.py
#
# Created:  Sep 2014, T. Orra
# Modified: Jan 2016, E. Botero

# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------

import numpy as np

# ----------------------------------------------------------------------
#  Calculate the range of  Payload Range Diagram
# ----------------------------------------------------------------------

def size_mission_range_given_weights(vehicle,mission,cruise_segment_tag,mission_payload,takeoff_weight=0.,reserve_fuel=0.):
    """ SUAVE.Methods.Performance.size_mission_range_given_weights(vehicle,mission,cruise_segment_tag,mission_payload,takeoff_weight=0.,reserve_fuel=0.):
        Calculates vehicle range for a given takeoff weight and payload

        Inputs:
            vehicle                     - SUave type vehicle
            mission                     - SUave type mission profile
            cruise_segment_tag          - Mission segment to be considered Cruise
            mission_payload             - float or 1d array with mission payload weight for each mission
            reserve_fuel   [optional]   - float of 1d array with required reserve fuel [kg]
            takeoff_weight [optional]   - float or 1d array with takeoff weight for each mission
                                          [if not informed, vehicle.mass_properties.m_takeoff is used]
        Outputs:
            distance              - float or 1d array with Range results for each mission
            fuel                  - float or 1d array with fuel burn results for each mission

        Assumptions:
            Constant altitude cruise.

    """
    #unpack
    masses = vehicle.mass_properties

    OEW = masses.operating_empty
    if not OEW:
        print "Error calculating Range for a Given TOW and Payload: Vehicle Operating Empty not defined"
        return True

    # Defining arrays for input and output
    mission_payload = np.atleast_1d(mission_payload)
    takeoff_weight  = np.atleast_1d(takeoff_weight)
    reserve_fuel    = np.atleast_1d(reserve_fuel)

    if not takeoff_weight[0]:
        takeoff_weight = np.multiply(np.ones_like(mission_payload),vehicle.mass_properties.max_takeoff)
        if not takeoff_weight[0]:
            print "Error calculating Range for a Given TOW and Payload: Vehicle takeoff weight not defined"
            return True

    # Check if # payload input is equal to # takeoff weights
    if len(mission_payload) == 1 and len(takeoff_weight) > 1:
        mission_payload = np.multiply(np.ones_like(takeoff_weight),mission_payload[0])

    # Check if # reserve_fuel input is equal to # takeoff weights
    if len(reserve_fuel) == 1 and len(takeoff_weight) > 1:
        reserve_fuel = np.multiply(np.ones_like(takeoff_weight),reserve_fuel[0])

    # Allocating results
    fuel     = np.zeros_like(mission_payload)
    distance = np.zeros_like(mission_payload)

    # Locate cruise segment to be variated
    for i in range(len(mission.segments)):          #loop for all segments
        if mission.segments[i].tag.upper() == cruise_segment_tag.upper() :
            segmentNum = i
            break
    print mission.segments
    TOW_ref = mission.segments[0].analyses.weights.mass_properties.takeoff 
    
    # Loop for range calculation of each input case
    for id,TOW in enumerate(takeoff_weight):
        PLD     =  mission_payload[id]
        FUEL    =  TOW - OEW - PLD - reserve_fuel[id]

        # Update mission takeoff weight
        vehicle.mass_properties.takeoff = TOW
        mission.segments[0].analyses.weights.mass_properties.takeoff = TOW

        # Evaluate mission with current TOW
        results = mission.evaluate()
        segment = results.segments[segmentNum]

        # Distance convergency in order to have total fuel equal to target fuel

        # User don't have the option of run a mission for a given fuel. So, we
        # have to iterate distance in order to have total fuel equal to target fuel

        maxIter  = 10    # maximum iteration limit
        tol      = 1.    # fuel convergency tolerance
        residual = 9999. # residual to be minimized
        iter     = 0     # iteration count

        while abs(residual) > tol and iter < maxIter:
            iter = iter + 1

            # Current total fuel burned in mission
            TotalFuel  = TOW - results.segments[-1].conditions.weights.total_mass[-1]

            # Difference between burned fuel and target fuel
            missingFuel = FUEL - TotalFuel

            # Current distance and fuel consuption in the cruise segment
            CruiseDist = segment.conditions.frames.inertial.position_vector[-1,0] - segment.conditions.frames.inertial.position_vector[0,0]                # Distance [m]
            CruiseFuel = segment.conditions.weights.total_mass[0] - segment.conditions.weights.total_mass[-1]    # [kg]
            # Current specific range (m/kg)
            CruiseSR    = CruiseDist / CruiseFuel        # [m/kg]

            # Estimated distance that will result in total fuel burn = target fuel
            DeltaDist  =  CruiseSR *  missingFuel
            mission.segments[segmentNum].distance = (CruiseDist + DeltaDist)

            # running mission with new distance
            results = mission.evaluate()
            segment = results.segments[segmentNum]

            # Difference between burned fuel and target fuel
            residual = ( TOW- results.segments[-1].conditions.weights.total_mass[-1] ) - FUEL

        # Allocating resulting range in ouput array.
        distance[id] = ( results.segments[-1].conditions.frames.inertial.position_vector[-1,0] ) #Distance [m]
        fuel[id] = FUEL

    mission.segments[0].analyses.weights.mass_properties.takeoff = TOW_ref

    return distance,fuel