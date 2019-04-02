## @ingroup Methods-Weights-Buildups-Common

# wiring.py
#
# Created: Jun, 2017, J. Smart
# Modified: Feb, 2018, J. Smart

#-------------------------------------------------------------------------------
# Imports
#-------------------------------------------------------------------------------

from SUAVE.Core import Units
from SUAVE.Components.Energy.Networks import Battery_Propeller
from SUAVE.Components.Energy.Networks import Lift_Forward_Propulsor
import numpy as np


#-------------------------------------------------------------------------------
# Wiring
#-------------------------------------------------------------------------------

## @ingroup Methods-Weights-Buildups-Common
def wiring(config,
           max_power_draw):
    """ weight = SUAVE.Methods.Weights.Buildups.Common.wiring(
            config,
            max_power_draw)
        
        Assumptions:
        Calculates mass of wiring required for a wing, including DC power
        cables and communication cables, assuming power cables run an average of
        half the fuselage length and height in addition to reaching the motor
        location on the wingspan, and that communication and sesor  wires run an
        additional length based on the fuselage and wing dimensions.

        Sources:
        Project Vahana Conceptual Trade Study

        Inputs:

            config                      SUAVE Config Data Structure
            max_power_draw              Maximum DC Power Draw           [W]

        Outputs:

            weight:                     Wiring Mass                     [kg]

    """

    #---------------------------------------------------------------------------
    # Unpack Inputs
    #---------------------------------------------------------------------------
    
    fLength     = config.fuselages.fuselage.lengths.total
    fHeight     = config.fuselages.fuselage.heights.maximum
    propulsor   = config.propulsors.propsulor

    #---------------------------------------------------------------------------
    # Sub-Function for Each Wing's Power Cables
    #---------------------------------------------------------------------------

    def wingCabling(wingspan, motor_spanwise_locations):

        nMotors = max(len(motor_spanwise_locations), 1)  # No. of motors on each half-wing, defaults to 1
        cablePower = max_power_draw / nMotors  # Power draw through each cable
        cableLength = 2 * (nMotors * (fLength / 2 + fHeight / 2) + np.sum(motor_spanwise_locations) * wingspan / 2)
        cableDensity = 1e-5
        massCables = cableDensity * cablePower * cableLength

        return np.array([massCables, cableLength])

    #----------------------------------------------------------------------------
    # Calculate Total Cable Mass and Length
    #----------------------------------------------------------------------------

    if isinstance(propulsor, Battery_Propeller):
        wingspan = fHeight
        motor_spanwise_locations = np.ones(int(propulsor.number_of_engines))
        massCables, cableLength = wingCabling(wingspan, motor_spanwise_locations)

    else:
        ML = np.array([0., 0.])

        for wing in config.wings:
            ML += wingCabling(wing.spans.projected, wing.motor_spanwise_locations)

        massCables = ML[0]
        cableLength = ML[1]

    #---------------------------------------------------------------------------
    # Determine mass of sensor/communication wires
    #---------------------------------------------------------------------------

    wiresPerBundle  = 6
    wireDensity     = 460e-5
    wireLength      = cableLength + (10 * fLength) +  wingspan
    massWires       = 2 * wireDensity * wiresPerBundle * wireLength

    #---------------------------------------------------------------------------
    # Sum Total
    #---------------------------------------------------------------------------

    weight = massCables + massWires

    return weight