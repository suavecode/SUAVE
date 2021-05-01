## @ingroup Methods-Weights-Buildups-Electric_Multicopter

# empty.py
#
# Created: Jun, 2017, J. Smart
# Modified: Apr, 2018, J. Smart
#           Mar 2020, M. Clarke

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

## @ingroup Methods-Weights-Buildups-Electric_Multicopter
def empty(config,settings,
          speed_of_sound                = 340.294,
          max_tip_mach                  = 0.65,
          disk_area_factor              = 1.15,
          max_thrust_to_weight_ratio    = 1.1,
          motor_efficiency              = 0.85 * 0.98):
    """ Calculates the empty fuselage mass for an electric helicopter including
        seats, avionics, servomotors, ballistic recovery system, rotor and hub
        assembly, transmission, and landing gear. Additionally incorporates
        results of the following common buildup scripts: 
            fuselage,py
            prop.py
            wing.py
            wiring.py

        Originally written as part of an AA 290 project inteded for trade study
        of the Electric Multicopter along with the following defined SUAVE vehicle types: 
            Electric Vectored_Thrust
            Electric Stopped Rotor
            
        Sources:
        Project Vahana Conceptual Trade Study
        https://github.com/VahanaOpenSource

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

    rRotor              = config.propulsors.vectored_thrust.rotor.tip_radius 
    rotor_bladeSol      = config.propulsors.vectored_thrust.rotor.blade_solidity    
    mBattery            = config.propulsors.vectored_thrust.battery.mass_properties.mass
    mPayload            = config.propulsors.vectored_thrust.payload.mass_properties.mass
    MTOW                = config.mass_properties.max_takeoff
     
    tipMach             = max_tip_mach
    k                   = disk_area_factor
    ToverW              = max_thrust_to_weight_ratio
    etaMotor            = motor_efficiency

    output.payload      = mPayload
    output.seats        = 30.
    output.avionics     = 15.
    output.motors       = config.propulsors.vectored_thrust.number_of_engines * 20.
    output.battery      = mBattery
    output.servos       = 5.2
    output.brs          = 16.
    output.hub          = MTOW * 0.04
    output.landing_gear = MTOW * 0.02

    #-------------------------------------------------------------------------------
    # Calculated Weights
    #-------------------------------------------------------------------------------

    # Preparatory Calculations
    rho_ref     = 1.225
    Vtip        = speed_of_sound * tipMach                     # Prop Tip Velocity
    omega       = Vtip/rRotor                                  # Prop Ang. Velocity
    maxThrust   = MTOW * ToverW * 9.8                          # Maximum Thrust
    Ct          = maxThrust/(rho_ref *np.pi*rRotor**2*Vtip**2) # Thrust Coefficient 
    AvgCL       = 6 * Ct / rotor_bladeSol                      # Average Blade CL
    AvgCD       = 0.012                                        # Average Blade CD
    V_AR        = 1.16*np.sqrt(maxThrust/(np.pi*rRotor**2))    # Autorotation Descent Velocity

    maxPower    = 1.15*maxThrust*(
                    k*np.sqrt(maxThrust/(2*rho_ref *np.pi*rRotor**2)) +
                    rotor_bladeSol*AvgCD/8*Vtip**3/(maxThrust/(rho_ref *np.pi*rRotor**2)))

    maxTorque = maxPower/omega

    # Component Weight Calculations

    output.rotor         = prop(config.propulsors.vectored_thrust.rotor,
                                maxThrust)
    output.tail_rotor    = prop(config.propulsors.vectored_thrust.rotor,
                                1.5*maxTorque/(1.25*rRotor))*0.2
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