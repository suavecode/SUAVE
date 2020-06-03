## @ingroup Methods-Weights-Buildups-Common

# wiring.py
#
# Created: Jun, 2017, J. Smart
# Modified: Feb, 2018, J. Smart
#           Mar 2020, M. Clarke

#-------------------------------------------------------------------------------
# Imports
#-------------------------------------------------------------------------------

from SUAVE.Core import Units
import numpy as np


#-------------------------------------------------------------------------------
# Wiring
#-------------------------------------------------------------------------------

## @ingroup Methods-Weights-Buildups-Common
def wiring(config,
           motor_spanwise_locations,
           max_power_draw):
    """ weight = SUAVE.Methods.Weights.Buildups.Common.wiring(
            config,
            motor_spanwise_locations,
            max_power_draw)
        
        Assumptions:
        Calculates mass of wiring required for a wing, including DC power
        cables and communication cables, assuming power cables run an average of
        half the fuselage length and height in addition to reaching the motor
        location on the wingspan, and that communication and sesor  wires run an
        additional length based on the fuselage and wing dimensions.

        Intended for use with the following SUAVE vehicle types, but may be used
        elsewhere:

            Electric Multicopter
            Electric Vectored_Thrust
            Electric Stopped Rotor

        Originally written as part of an AA 290 project intended for trade study
        of the above vehicle types.
        
        Sources:
        Project Vahana Conceptual Trade Study

        Inputs:

            config                      SUAVE Config Data Structure
            motor_spanwise_locations    Motor Semi-Span Fractions       [Unitless]
            max_power_draw              Maximum DC Power Draw           [W]

        Outputs:

            weight:                     Wiring Mass                     [kg]

    """

    #---------------------------------------------------------------------------
    # Unpack Inputs
    #---------------------------------------------------------------------------
    
    fLength     = config.fuselages.fuselage.lengths.total
    fHeight     = config.fuselages.fuselage.heights.maximum
    wingspan    = config.wings['main_wing'].spans.projected

    nMotors = max(len(motor_spanwise_locations),1)    # No. of motors on each half-wing, defaults to 1

    #---------------------------------------------------------------------------
    # Determine mass of Power Cables
    #---------------------------------------------------------------------------

    cablePower      = max_power_draw/nMotors      # Power draw through each cable
    cableLength     = 2 * (nMotors * (fLength/2 + fHeight/2) + np.sum(motor_spanwise_locations) * wingspan/2)
    cableDensity    = 1e-5
    massCables      = cableDensity * cablePower * cableLength

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