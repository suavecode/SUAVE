## @ingroup Methods-Weights-Buildups-Electric_Helicopter

# empty.py
#
# Created: Jun, 2017, J. Smart
# Modified: Apr, 2018, J. Smart

#-------------------------------------------------------------------------------
# Imports
#-------------------------------------------------------------------------------

from SUAVE.Core import Units, Data
from SUAVE.Methods.Weights.Buildups.Common.fuselage import fuselage
from SUAVE.Methods.Weights.Buildups.Common.prop import prop
from SUAVE.Methods.Weights.Buildups.Common.wiring import wiring
import numpy as np

#-------------------------------------------------------------------------------
# Empty
#-------------------------------------------------------------------------------

## @ingroup Methods-Weights-Buildups-Electric_Helicopter
def empty(config,
          speed_of_sound                = 340.294,
          max_tip_mach                  = 0.65,
          disk_area_factor              = 1.15,
          max_thrust_to_weight_ratio    = 1.1,
          motor_efficiency              = 0.85 * 0.98):
    """weight = SUAVE.Methods.Weights.Buildups.Electric_Helicopter.empty(
            config,
            speed_of_sound              = 340.294,
            maximumTipMach              = 0.65,
            disk_area_factor            = 1.15,
            max_thrust_to_weight_ratio  = 1.1,
            motor_efficiency            = 0.85 * 0.98)

        Calculates the empty fuselage mass for an electric helicopter including
        seats, avionics, servomotors, ballistic recovery system, rotor and hub
        assembly, transmission, and landing gear. Additionally incorporates
        results of the following common buildup scripts:

            fuselage,py
            prop.py
            wing.py
            wiring.py

        Originally written as part of an AA 290 project inteded for trade study
        of the Electric Helicopter along with the following defined SUAVE vehicle types:

            Electric Tiltrotor
            Electric Stopped Rotor
            
        Sources:
        Project Vahana Conceptual Trade Study

        Inputs:

            config                          SUAVE Config Data Structure
            speed_of_sound                  Local Speed of Sound                [m/s]
            max_tip_mach                    Allowable Tip Mach Number           [Unitless]
            disk_area_factor                Inverse of Disk Area Efficiency     [Unitless]
            max_thrust_to_weight_ratio      Allowable Thrust to Weight Ratio    [Unitelss]
            motor_efficiency                Motor Efficiency                    [Unitless]

        Outputs:

            output:                         Data Dictionary of Component Masses       [kg]

    """

    output = Data()

#-------------------------------------------------------------------------------
# Unpack Inputs
#-------------------------------------------------------------------------------

    rProp               = config.propulsors.network.propeller.prop_attributes.tip_radius
    mBattery            = config.propulsors.network.battery.mass_properties.mass
    mPayload            = config.propulsors.network.payload.mass_properties.mass
    MTOW                = config.mass_properties.max_takeoff
    fLength             = config.fuselages.fuselage.lengths.total
    fWidth              = config.fuselages.fuselage.width
    fHeight             = config.fuselages.fuselage.heights.maximum
    
    sound               = speed_of_sound
    tipMach             = max_tip_mach
    k                   = disk_area_factor
    ToverW              = max_thrust_to_weight_ratio
    etaMotor            = motor_efficiency

    output.payload      = mPayload
    output.seats        = 30.
    output.avionics     = 15.
    output.motors       = config.propulsors.network.number_of_engines * 20.
    output.battery      = mBattery
    output.servos       = 5.2
    output.brs          = 16.
    output.hub          = MTOW * 0.04
    output.landing_gear = MTOW * 0.02

#-------------------------------------------------------------------------------
# Calculated Weights
#-------------------------------------------------------------------------------

    # Preparatory Calculations

    Vtip        = sound * tipMach                           # Prop Tip Velocity
    omega       = Vtip/rProp                                # Prop Ang. Velocity
    maxThrust   = MTOW * ToverW * 9.8                       # Maximum Thrust
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

    output.rotor         = prop(config.propulsors.network.propeller,
                                maxThrust)
    output.tail_rotor    = prop(config.propulsors.network.propeller,
                                1.5*maxTorque/(1.25*rProp))*0.2
    output.transmission  = maxPower * 1.5873e-4          # From NASA OH-58 Study
    output.fuselage      = fuselage(config)
    output.wiring        = wiring(config,np.ones(8),maxPower/etaMotor)

#-------------------------------------------------------------------------------
# Weight Summations
#-------------------------------------------------------------------------------

    output.structural = (output.rotor +
                        output.hub +
                        output.fuselage +
                        output.landing_gear
                        ) * Units.kg

    output.empty = 1.1 * (
                        output.structural +
                        output.seats +
                        output.avionics +
                        output.battery +
                        output.motors +
                        output.servos +
                        output.wiring +
                        output.brs
                        ) * Units.kg
    
    output.total = 1.1 * (
                        output.structural +
                        output.payload +
                        output.avionics +
                        output.battery +
                        output.motors +
                        output.servos +
                        output.wiring +
                        output.brs
                        ) * Units.kg
    return output