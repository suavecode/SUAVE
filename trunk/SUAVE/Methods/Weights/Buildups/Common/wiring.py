## @ingroup Methods-Weights-Buildups-Common

# wiring.py
#
# Created: Jun, 2017, J. Smart
# Modified: Feb, 2018, J. Smart
#           Mar 2020, M. Clarke
#           May 2021, M. Clarke

#-------------------------------------------------------------------------------
# Imports
#-------------------------------------------------------------------------------

from SUAVE.Core import Units
import numpy as np


#-------------------------------------------------------------------------------
# Wiring
#-------------------------------------------------------------------------------

## @ingroup Methods-Weights-Buildups-Common
def wiring(wing, config, cablePower):
    """ weight = SUAVE.Methods.Weights.Buildups.Common.wiring(
            wing,
            config, 
            cablePower)
        
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
    if 'motor_spanwise_locations' in config.wings[wing.tag]:
        fLength     = config.fuselages.fuselage.lengths.total
        fHeight     = config.fuselages.fuselage.heights.maximum
        MSL         = config.wings[wing.tag].motor_spanwise_locations
        wingspan    = wing.spans.projected 
        nMotors     = max(len(MSL),1)    # No. of motors on each half-wing, defaults to 1
        
        #---------------------------------------------------------------------------
        # Determine mass of Power Cables
        #--------------------------------------------------------------------------- 
        cableLength     = (nMotors * (fLength/2 + fHeight/2)) + (np.sum(abs(MSL)) * wingspan/2) 
        cableDensity    = 5.7e-6
        massCables      = cableDensity * cablePower * cableLength
        
        #---------------------------------------------------------------------------
        # Determine mass of sensor/communication wires
        #---------------------------------------------------------------------------
        
        wiresPerBundle  = 6
        wireDensity     = 460e-5
        wireLength      = cableLength + (10 * fLength) +  4*wingspan
        massWires       = 2 * wireDensity * wiresPerBundle * wireLength
        
        #---------------------------------------------------------------------------
        # Sum Total
        #---------------------------------------------------------------------------
        
        weight = massCables + massWires
    else:
        weight = 0.0
    return weight