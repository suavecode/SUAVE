## @ingroup Methods-Performance
# electric_payload_range.py
#
# Created:  Dec 2020, J. Smart
# Modified:

#------------------------------------------------------------------------------
# Imports
#------------------------------------------------------------------------------

from SUAVE.Core import Units, Data

import numpy as np
import matplotlib.pyplot as plt

#------------------------------------------------------------------------------
# Determine Max Payload Range and Ferry Range
#------------------------------------------------------------------------------

def electric_payload_range(vehicle,
                           mission,
                           cruise_segment_tag,
                           display_plot=True):

    """Calculates and plots a payload range diagram for an electric vehicle.

    Assumptions:

    Vehicle is defined with some combination of weight characteristics that
    allow determination of maximum payload and zero-payload weight (Empty
    weight, max takeoff, and max payload weight). Mission specified includes
    an electric variable distance cruise segment.

    Source:

        N/A

    Inputs:

        vehicle.mass_properties.
            operating_empty                                     [kg]
            max_payload                                         [kg]
            max_takeoff                                         [kg]

        mission.segments.
            cruise.variable_cruise_distance                     [Converge SOC]
                .tag                                            [String]
            analyses.weights.vehicle.mass_properties.takeoff    [kg]

        cruise_segment_tag:                                     [String Match to Above]

        reserve_soc:                                            [0-1, Percentage]

        display_plot                                            [True/False]

    Outputs:

        payload_range.
            range                                               [m]
            payload                                             [m]
            takeoff_weight                                      [kg]

    Properties Used:

        N/A
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

    for i in range(2):
        mission.segments[0].analyses.weights.vehicle.mass_properties.takeoff = TOW[i]
        results = mission.evaluate()
        segment = results.segments[cruise_segment_tag]
        R[i]    = results.segments[-1].conditions.frames.inertial.position_vector[-1,0]

    # Insert Starting Point for Diagram Construction

    R.insert(0, 0)
    PLD.insert(0, MaxPLD)
    TOW.insert(0, 0)

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