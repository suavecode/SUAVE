# wiring.py
#
# Created: Jun 2017, J. Smart

#-------------------------------------------------------------------------------
# Imports
#-------------------------------------------------------------------------------

from SUAVE.Core import Units
from SUAVE.Attributes import Solids
import numpy as np


#-------------------------------------------------------------------------------
# Wiring
#-------------------------------------------------------------------------------

def wiring(fLength, fHeight, wingspan, xMotor, P_max):
    """ weight = SUAVE.Methods.Weights.Correlations.eHelicopter.wiring(
            fLength,
            fHeight,
            wingspan,
            xMotor,
            P_max
        )

        Calculates mass of wiring required for a wing, including DC power
        cables and communication cables, assuming power cables run an average of
        half the fuselage length and height in addition to reaching the motor
        location on the wingspan, and that communication and sesor  wires run an
        additional length based on the fuselage and wing dimensions.

        Intended for use with the following SUAVE vehicle types, but may be used
        elsewhere:

            eHelicopter
            eTiltwing
            eTiltrotor
            eStopped_Rotor

        Originally written as part of an AA 290 project intended for trade study
        of the above vehicle types.

        Inputs:

            fLength:    Fuselage Length             [m]
            fHeight:    Fuselage Height             [m]
            wingspan:   Wingspan                    [m]
            xMotor:     Motor Semi-Span Fractions   [Unitless]
            P_max:      Maximum DC Power Draw       [W]

        Outputs:

            weight:     Wiring Mass               [kg]

    """

    nMotors = max(len(xMotor),1)    # No. of motors on each half-wing, defaults to 1

# Determine mass of Power Cables

    cablePower      = P_max/nMotors      # Power draw through each cable
    cableLength     = 2 * (nMotors * (fLength/2 + fHeight/2) + np.sum(xMotor) * wingspan/2)
    cableDensity    = 1e-5
    massCables      = cableDensity * cablePower * cableLength

# Determine mass of sensor/communication wires

    wiresPerBundle  = 6
    wireDensity     = 460e-5
    wireLength      = cableLength + (10 * fLength) +  wingspan
    massWires       = 2 * wireDensity * wiresPerBundle * wireLength

    weight = massCables + massWires

    return weight