# empty.py
#
# Created: Jun 2017, J. Smart

#-------------------------------------------------------------------------------
# Imports
#-------------------------------------------------------------------------------

from SUAVE.Core import Units, Data
from SUAVE.Attributes import Solids
from fuselage import fuselage
from prop import prop
from wiring import wiring
import numpy as np

#-------------------------------------------------------------------------------
# Empty
#-------------------------------------------------------------------------------

def empty(vehicle):
    """weight = SUAVE.Methods.Weights.Correlations.eStopped_Rotor.empty(
            rLiftProp,
            rThrustProp,
            mBattery,
            mMotors,
            mPayload,
            MTOW,
            nLiftRotors,
            nThrustRotors,
            liftBlades,
            thrustBlades,
            fLength,
            fWidth,
            fHeight
        )

        Calculates the empty fuselage mass for an electric stopped rotor including
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

            rLiftProp:     Lift Propeller Radius           [m]
            rThrustProp:   Thrust Propeller Radius         [m]
            mBattery:      Battery Mass                    [m]
            mMotors:       Total Motor Mass                [m]
            mPayload:      Payload Mass                    [m]
            MTOW:          Maximum TO Weight               [N]
            nLiftProps:    Number of Lift Propellers       [Unitless]
            nThrustProps:  Number of Thrust                [Unitless]
            nLiftBlades:   Number of Lift Prop Blades      [Unitless]
            nThrustBlades: Number of Thrust Prop Blades    [Unitless]
            fLength:       Fuselage Length                 [m]
            fWidth:        Fuselage Width                  [m]
            fHeight:       Fuselage Height                 [m]

        Outputs:

            weight:        Dictionary of Component Masses  [m]

    """

    output = Data()

#-------------------------------------------------------------------------------
# Assumed Weights
#-------------------------------------------------------------------------------

    ouptut.payload          = vehicle.net.payload.mass_properties.mass
    output.seats            = 30.
    output.avionics         = 15.
    output.motors           = 10 * np.ceil(vehicle.mass_properties.max_takeoff * (1/200. + 1/5.))
    output.battery          = vehicle.net.battery.mass_properties.mass
    output.servos           = 0.65 * np.ceil(vehicle.mass_properties.max_takeoff * (1/200. + 1/5.))
    output.brs              = 16.
    output.hubs             = 2 * np.ceil(vehicle.mass_properties.max_takeoff * (1/200. + 1/5.))
    output.landing_gear     = vehicle.mass_properties.max_takeoff * 0.02

#-------------------------------------------------------------------------------
# Calculated Weights
#-------------------------------------------------------------------------------

    # Preparatory Calculations

    sound       = 340.294       # Speed of Sound
    tipMach     = 0.65          # Propeller Tip Mach limit
    k           = 1.15          # Effective Disk Area Factor
    ToverW      = 1.1           # Thrust over MTOW
    etaMotor    = 0.85 * 0.98   # Collective Motor and Gearbox Efficiencies

    Vtip        = sound * tipMach                               # Prop Tip Velocity
    omega       = Vtip/0.8                                      # Prop Ang. Velocity
    maxLift     = vehicle.mass_properties.max_takeoff * ToverW  # Maximum Thrust
    Ct          = maxLift/(1.225*np.pi*0.8**2*Vtip**2)          # Thrust Coefficient
    bladeSol    = 0.1                                           # Blade Solidity
    AvgCL       = 6 * Ct / bladeSol                             # Average Blade CL
    AvgCD       = 0.012                                         # Average Blade CD
   
    maxLiftPower    = 1.15*maxLift*(
                    k*np.sqrt(maxLift/(2*1.225*np.pi*0.8**2)) +
                    bladeSol*AvgCD/8*Vtip**3/(maxLift/(1.225*np.pi*0.8**2))
                    )

    maxTorque = maxLiftPower/omega

    # Component Weight Calculations

    output.lift_rotors      = prop(0.8, maxLift, 4)
    output.thrust_rotors    = prop(0.8, maxLift/5, 2)
    output.fuselage         = fuselage(vehicle.fuselage.lengths.total,
                                       vehicle.fuselage.width,
                                       vehicle.fuselage.heights.maximum,
                                       vehicle.wings['Main Wing'].spans.projected,
                                       vehicle.mass_properties.max_takeoff)
    output.wiring           = wiring(vehicle.fuselage.lengths.total,
                                     vehicle.fuselage.heights.maximum,
                                     vehicle.wings['Main Wing'].spans.projected,
                                     np.ones(8)**0.25,
                                     maxLiftPower/etaMotor)

#-------------------------------------------------------------------------------
# Weight Summations
#-------------------------------------------------------------------------------

    output = Data()

    output.structural   = (output.lift_rotors +
                            output.thrust_rotors +
                            output.hubs +
                            output.fuselage +
                            output.landing_gear
                            )

    output.empty        = 1.1 * (
                            output.structural +
                            output.seats +
                            output.avionics +
                            output.battery +
                            output.motors +
                            output.servos +
                            output.wiring +
                            output.brs
                            )
    
    output.total        = (output.empty +
                            output.payload) 

    return output