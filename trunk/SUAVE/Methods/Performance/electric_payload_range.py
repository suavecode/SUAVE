## @ingroup Methods-Performance
# electric_payload_range.py
#
# Created: Jan 2021, J. Smart
# Modified:

#------------------------------------------------------------------------------
# Imports
#------------------------------------------------------------------------------

from SUAVE.Core import Units, Data

import numpy as np
import matplotlib.pyplot as plt

#------------------------------------------------------------------------------
# Electric Payload Range Function
#------------------------------------------------------------------------------

## @ingroup Methods-Performance
def electric_payload_range(vehicle,
                           mission,
                           cruise_segment_tag,
                           display_plot=True):

    """electric_payload_range(vehicle,
                           mission,
                           cruise_segment_tag,
                           display_plot=True):

        Calculates and optionally displays a payload range diagram for a
        Variable Cruise Distance - State of Charge SUAVE Mission and Vehicle.

        Sources:
        N/A

        Assumptions:

        Assumes use of Battery Propeller Energy Network

        Inputs:

            vehicle                         SUAVE Vehicle Structure
                .mass_properties            SUAVE Mass Properties Structure
                    .operating_empty        Vehicle Operating Empty Mass    [kg]
                    .max_payload            Vehicle Maximum Payload Mass    [kg]
                    .max_takeoff            Vehicle Maximum Takeoff Mass    [kg]

            mission                         SUAVE Mission Structure
                .Variable_Range_Cruise      Mission Type
                    .Given_State_of_Charge  Convergence Criteria
                .cruise_tag                 Mission Segment Tag             [String]
                .target_state_of_charge     End Mission State of Charge     [Unitless]

            cruise_segment_tag              mission.cruise_tag              [String]

        Outputs:

            payload_range = Data()
                .range                      [0, Max PLD Range, Ferry Range] [m]
                .payload                    [Max PLD, Max PLD , 0]          [kg]
                .takeoff_weight             [MTOW, MTOW, OEW]               [kg]
    """

    # Unpack Weights

    masses = vehicle.mass_properties

    if not masses.operating_empty:
        print("Error calculating Payload Range Diagram: vehicle Operating Empty Weight is undefined.")
        return True
    else:
        OEW = masses.operating_empty

    if not masses.max_payload:
        print("Error calculating Payload Range Diagram: vehicle Maximum Payload Weight is undefined.")
        return True
    else:
        MaxPLD = masses.max_payload

    if not masses.max_takeoff:
        print("Error calculating Payload Range Diagram: vehicle Maximum Payload Weight is undefined.")
        return True
    else:
        MTOW = masses.max_takeoff

    # Define Diagram Points
    # Point = [Value at Maximum Payload Range,  Value at Ferry Range]

    TOW =   [MTOW,      OEW]    # Takeoff Weights
    PLD =   [MaxPLD,    0.]     # Payload Weights

    # Initialize Range Array

    R = np.zeros(2)

    # Calculate Vehicle Range for Max Payload and Ferry Conditions

    for i in range(2):
        mission.segments[0].analyses.weights.vehicle.mass_properties.takeoff = TOW[i]
        results = mission.evaluate()
        segment = results.segments[cruise_segment_tag]
        R[i]    = segment.conditions.frames.inertial.position_vector[-1,0]

    # Insert Starting Point for Diagram Construction

    R = np.insert(R, 0, 0)
    PLD = np.insert(PLD, 0, MaxPLD)
    TOW = np.insert(TOW, 0, 0)

    # Pack Results

    payload_range = Data()
    payload_range.range             = R
    payload_range.payload           = PLD
    payload_range.takeoff_weight    = TOW

    if display_plot:

        plt.plot(R, PLD, 'r')
        plt.xlabel('Range (m)')
        plt.ylabel('Payload (kg)')
        plt.title('Payload Range Diagram')
        plt.grid(True)
        plt.show()

    return payload_range
