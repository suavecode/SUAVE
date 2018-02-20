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

def empty(config,
          speedOfSound       = 340.294,
          maximumTipMach     = 0.65,
          diskAreaFactor     = 1.15,
          maxThrustToWeight  = 1.1,
          motorEfficiency    = 0.85 * 0.98):
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
        of the eHelicotor along with the following defined SUAVE config types:

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
# Unpack Inputs
#-------------------------------------------------------------------------------

    rProp               = config.propulsors.network.propeller.prop_attributes.tip_radius
    mBattery            = config.propulsors.network.battery.mass_properties.mass
    mPayload            = config.propulsors.network.payload.mass_properties.mass
    MTOW                = config.mass_properties.max_takeoff
    nLiftProps          = config.propulsors.network.number_of_engines/2
    nThrustProps        = config.propulsors.network.number_of_engines/2
    nLiftBlades         = config.propulsors.network.propeller.prop_attributes.number_blades
    nThrustBlades       = config.propulsors.network.propeller.prop_attributes.number_blades
    fLength             = config.fuselages.fuselage.lengths.total
    fWidth              = config.fuselages.fuselage.width
    fHeight             = config.fuselages.fuselage.heights.maximum
    maxSpan             = config.wings['main_wing'].spans.projected
    
    sound               = speedOfSound
    tipMach             = maximumTipMach
    k                   = diskAreaFactor
    ToverW              = maxThrustToWeight
    etaMotor            = motorEfficiency

    output.payload          = mPayload * Units.kg
    output.seats            = 30. *Units.kg
    output.avionics         = 15. *Units.kg
    output.motors           = config.propulsors.network.number_of_engines * 10. *Units.kg
    output.battery          = mBattery *Units.kg
    output.servos           = config.propulsors.network.number_of_engines * 0.65 *Units.kg
    output.brs              = 16. *Units.kg
    output.hubs             = config.propulsors.network.number_of_engines * 2. *Units.kg
    output.landing_gear     = MTOW * 0.02 *Units.kg

#-------------------------------------------------------------------------------
# Calculated Weights
#-------------------------------------------------------------------------------

    # Preparatory Calculations

    Vtip        = sound * tipMach                               # Prop Tip Velocity
    omega       = Vtip/0.8                                      # Prop Ang. Velocity
    maxLift     = config.mass_properties.max_takeoff * ToverW   # Maximum Thrust
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

    output.lift_rotors      = (prop(config.propulsors.network.propeller, maxLift, 4) 
                               * (len(config.wings['main_wing'].xMotor) 
                                  + len(config.wings['main_wing'].xMotor))) *Units.kg
    output.thrust_rotors    = prop(config.propulsors.network.propeller, maxLift/5, 2) *Units.kg
    output.fuselage         = fuselage(config) *Units.kg
    output.wiring           = wiring(config,
                                     np.ones(8)**0.25,
                                     maxLiftPower/etaMotor) *Units.kg
    output.main_wing = wing(config.wings['main_wing'],
                            config, 
                            maxLift/5) *Units.kg
    output.sec_wing = wing(config.wings['secondary_wing'],
                            config,
                            maxLift/5) *Units.kg
    
    
#-------------------------------------------------------------------------------
# Weight Summations
#-------------------------------------------------------------------------------


    output.structural   = (output.lift_rotors +
                            output.thrust_rotors +
                            output.hubs +
                            output.fuselage + 
                            output.landing_gear +
                            output.main_wing +
                            output.sec_wing
                            ) *Units.kg

    output.empty        = 1.1 * (
                            output.structural +
                            output.seats +
                            output.avionics +
                            output.battery +
                            output.motors +
                            output.servos +
                            output.wiring +
                            output.brs
                            ) *Units.kg
    
    output.total        = (output.empty +
                            output.payload) *Units.kg

    return output