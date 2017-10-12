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
from wing import wing
import numpy as np

#-------------------------------------------------------------------------------
# Empty
#-------------------------------------------------------------------------------

def empty(vehicle):
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

    output.payload          = vehicle.propulsors.network.payload.mass_properties.mass
    output.seats            = 30.
    output.avionics         = 15.
    output.motors           = 10 * np.ceil(vehicle.mass_properties.max_takeoff * (1/200. + 1/5.))
    output.battery          = vehicle.propulsors.network.battery.mass_properties.mass
    output.servos           = 0.65 * np.ceil(vehicle.mass_properties.max_takeoff * (1/200. + 1/5.))
    output.rotor_servos     = 2 * (len(vehicle.wings['main wing'].xMotor) + len(vehicle.wings['secondary wing'].xMotor))
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

    output.lift_rotors      = prop(0.8, maxLift, 4) * (len(vehicle.wings['main wing'].xMotor) + len(vehicle.wings['secondary wing'].xMotor))
    output.fuselage         = fuselage(vehicle.fuselages['fuselage'].lengths.total,
                                       vehicle.fuselages['fuselage'].width,
                                       vehicle.fuselages['fuselage'].heights.maximum,
                                       vehicle.wings['main_wing'].spans.projected,
                                       vehicle.mass_properties.max_takeoff)
    output.wiring           = wiring(vehicle.fuselages['fuselage'].lengths.total,
                                     vehicle.fuselages['fuselage'].heights.maximum,
                                     vehicle.wings['main_wing'].spans.projected,
                                     np.ones(8)**0.25,
                                     maxLiftPower/etaMotor)
    output.main_wing = wing(vehicle.mass_properties.max_takeoff, 
                            vehicle.wings['main wing'].spans.projected,
                            vehicle.wings['main wing'].chords.mean_aerodynamic,
                            vehicle.wings['main wing'].thickness_to_chord, 
                            vehicle.wings['main wing'].winglet_fraction, 
                            vehicle.wings['main wing'].areas.reference/(vehicle.wings['main wing'].areas.reference + vehicle.wings['secondary wing'].areas.reference),
                            vehicle.wings['main wing'].xMotor, 
                            maxLift/5)
    output.sec_wing = wing(vehicle.mass_properties.max_takeoff, 
                            vehicle.wings['secondary wing'].spans.projected,
                            vehicle.wings['secondary wing'].chords.mean_aerodynamic,
                            vehicle.wings['secondary wing'].thickness_to_chord, 
                            vehicle.wings['secondary wing'].winglet_fraction, 
                            vehicle.wings['secondary wing'].areas.reference/(vehicle.wings['main wing'].areas.reference + vehicle.wings['secondary wing'].areas.reference),
                            vehicle.wings['secondary wing'].xMotor, 
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