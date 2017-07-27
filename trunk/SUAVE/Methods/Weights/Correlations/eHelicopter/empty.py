# empty.py
#
# Created: Jun 2017, J. Smart

#-------------------------------------------------------------------------------
# Imports
#-------------------------------------------------------------------------------

from SUAVE.Core import Units
from SUAVE.Attributes import Solids
from fuselage import fuselage
from prop import prop
from wiring import wiring
import numpy as np

#-------------------------------------------------------------------------------
# Empty
#-------------------------------------------------------------------------------

def empty(rProp, mBattery, mMotors, mPayload, MTOW):
    """weight = SUAVE.Methods.Weights.Correlations.eHelicopter.empty(
            rProp,
            mBattery,
            mMotors,
            mPayload,
            MTOW,
        )

        Calculates the empty fuselage mass for an electric helicopter including
        seats, avionics, servomotors, ballistic recovery system, rotor and hub
        assembly, transmission, and landing gear. Additionally incorporates
        results of the following correlation scripts:

            fuselage,py
            prop.py
            wing.py
            wiring.py

        Originally written as part of an AA 290 project inteded for trade study
        of the eHelicotor along with the following defined SUAVE vehicle types:

            eTiltwing
            eTiltrotor
            eStopped_Rotor

        Inputs:

            rProp:      Propeller Radius                [m]
            mBattery:   Battery Mass                    [m]
            mMotors:    Total Motor Mass                [m]
            mPayload:   Payload Mass                    [m]
            MTOW:       Maximum TO Weight               [N]
            propBlades: Number of Propeller Blades      [Dimensionless]
            tailBlades: Number of Tail Rotor Blades     [Dimensionless]
            fLength:    Fuselage Length                 [m]
            fWidth:     Fuselage Width                  [m]
            fHeight:    Fuselage Height                 [m]

        Outputs:

            weight:     Dictionary of Component Masses  [m]

    """

    weight = {}

#-------------------------------------------------------------------------------
# Assumed Weights
#-------------------------------------------------------------------------------

    weight['Payload']       = mPayload
    weight['Seats']         = 30.
    weight['Avionics']      = 15.
    weight['Motors']        = mMotors
    weight['Battery']       = mBattery
    weight['Servos']        = 5.2
    weight['BRS']           = 16.
    weight['Hub']           = MTOW * 0.04
    weight['Landing Gear']  = MTOW * 0.02

#-------------------------------------------------------------------------------
# Calculated Weights
#-------------------------------------------------------------------------------

    # Preparatory Calculations

    sound       = 340.294       # Speed of Sound
    tipMach     = 0.65          # Propeller Tip Mach limit
    k           = 1.15          # Effective Disk Area Factor
    ToverW      = 1.1           # Thrust over MTOW
    etaMotor    = 0.85 * 0.98   # Collective Motor and Gearbox Efficiencies

    Vtip        = sound * tipMach                           # Prop Tip Velocity
    omega       = Vtip/rProp                                # Prop Ang. Velocity
    maxThrust   = MTOW * ToverW                             # Maximum Thrust
    Ct          = maxThrust/(1.225*np.pi*rProp**2*Vtip**2)  # Thrust Coefficient
    bladeSol    = 0.1                                       # Blade Solidity
    AvgCL       = 6 * Ct / bladeSol                         # Average Blade CL
    AvgCD       = 0.012                                     # Average Blade CD
    V_AR        = 1.16*np.sqrt(maxThrust/(np.pi*rProp**2))  # Autorotation Descent Velocity

    maxPower    = 1.15*maxThrust*(
                    k*np.sqrt(maxThrust/(2*1.225*np.pi*rProp**2)) +
                    bladeSol*AvgCD/8*Vtip**3/(maxThrust/(1.225*np.pi*rProp**2))
                    )

    maxTorque = maxPower/omega

    # Component Weight Calculations

    weight['Rotor']         = prop(rProp, maxThrust, propBlades)
    weight['Tail Rotor']    = prop(rProp/5, 1.5*maxTorque/(1.25*rProp), tailBlades)
    weight['Transmission']  = maxPower * 1.5873e-4      # From NASA OH-58 Study
    weight['Fuselage']      = fuselage(fLength,fWidth,fHeight,0,MTOW)
    weight['Wiring']        = wiring(fLength,fHeight,1,fLength/2,np.ones(8),maxPower/etaMotor)

#-------------------------------------------------------------------------------
# Weight Summations
#-------------------------------------------------------------------------------

    weight['Structural'] = (weight['Rotor'] +
                            weight['Hub'] +
                            weight['Fuselage'] +
                            weight['Landing Gear']
                            )

    weight['Total'] = 1.1 * (
                        weight['Structural'] +
                        weight['Payload'] +
                        weight['Seat'] +
                        weight['Avionics'] +
                        weight['Battery'] +
                        weight['Motors'] +
                        weight['Servos'] +
                        weight['Wiring'] +
                        weight['BRS']
                        )

    return weight
