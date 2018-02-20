# empty.py
#
# Created: Jun 2017, J. Smart

#-------------------------------------------------------------------------------
# Imports
#-------------------------------------------------------------------------------

from SUAVE.Core import Units, Data
from SUAVE.Attributes.Solids import (
    BiCF, Honeycomb, Paint, UniCF, Acrylic, Steel, Aluminum, Epoxy, Nickel, Rib)
from SUAVE.Methods.Weights.Buildups.Common.fuselage import fuselage
from SUAVE.Methods.Weights.Buildups.Common.prop import prop
from SUAVE.Methods.Weights.Buildups.Common.wiring import wiring
from SUAVE.Methods.Weights.Buildups.Common.wing import wing
import numpy as np

#-------------------------------------------------------------------------------
# Empty
#-------------------------------------------------------------------------------

def empty(config):
    """weight = SUAVE.Methods.Weights.Buildups.eTiltrotor.empty(
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

        Calculates the empty fuselage mass for an electric tiltrotor including
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
            nThrustProps:  Number of Thrust Propellers     [Unitless]
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

    output.payload          = config.propulsors.network.payload.mass_properties.mass
    output.seats            = 30.
    output.avionics         = 15.
    output.motors           = 10 * config.propulsors.network.number_of_engines
    output.battery          = config.propulsors.network.battery.mass_properties.mass
    output.servos           = 0.65 * config.propulsors.network.number_of_engines
    output.rotor_servos     = 2 * (len(config.wings['main_wing'].xMotor) + len(config.wings['main_wing'].xMotor))
    output.brs              = 16.
    output.hubs             = 2 * config.propulsors.network.number_of_engines
    output.landing_gear     = config.mass_properties.max_takeoff * 0.02

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
    maxLift     = config.mass_properties.max_takeoff * ToverW  # Maximum Thrust
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

    output.lift_rotors      = prop(config.propulsors.network.propeller, maxLift, 4) * (len(config.wings['main_wing'].xMotor) + len(config.wings['main_wing'].xMotor))
    output.fuselage         = fuselage(config)
    output.wiring           = wiring(config,
                                     np.ones(8)**0.25,
                                     maxLiftPower/etaMotor)
    output.main_wing = wing(config.wings['main_wing'],
                            config,
                            maxLift/5)
    output.sec_wing = wing(config.wings['secondary_wing'],
                            config,
                            maxLift/5)
    
    
#-------------------------------------------------------------------------------
# Weight Summations
#-------------------------------------------------------------------------------


    output.structural   = (output.lift_rotors +
                            output.hubs +
                            output.fuselage + 
                            output.landing_gear +
                            output.main_wing +
                            output.sec_wing
                            )

    output.empty        = 1.1 * (
                            output.structural +
                            output.seats +          
                            output.avionics +
                            output.battery +
                            output.motors +
                            output.servos +
                            output.rotor_servos +
                            output.wiring +
                            output.brs
                            )
    
    output.total        = (output.empty +
                            output.payload) 

    return output