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

#------------------------------------------------------------------------------
# Electric Payload Range Function
#------------------------------------------------------------------------------

def electric_payload_range(vehicle,
                           mission,
                           cruise_segment_tag,
                           display_plot=True):

    '''
    TODO: Add docstring
    '''

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
