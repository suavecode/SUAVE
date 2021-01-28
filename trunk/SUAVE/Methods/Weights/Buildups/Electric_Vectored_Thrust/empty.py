## @ingroup Methods-Weights-Buildups-Electric_Vectored_Thrust
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
from SUAVE.Methods.Weights.Buildups.Common.wing import wing
import numpy as np

#-------------------------------------------------------------------------------
# Empty
#-------------------------------------------------------------------------------

## @ingroup Methods-Weights-Buildups-Electric_Vectored_Thrust
def empty(config,
          settings,
          speed_of_sound                = 340.294,
          max_tip_mach                  = 0.65,
          disk_area_factor              = 1.15,
          max_thrust_to_weight_ratio    = 1.1,
          motor_efficiency              = 0.85 * 0.98):
    """weight = SUAVE.Methods.Weights.Buildups.electricVectored_Thrust.empty(
            config,
            speed_of_sound              = 340.294,
            max_tip_mach                = 0.65,
            disk_area_factor            = 1.15,
            max_thrust_to_weight_ratio  = 1.1,
            motor_efficiency            = 0.85 * 0.98)
        
        Calculates the empty fuselage mass for an electric tiltrotor including
        seats, avionics, servomotors, ballistic recovery system, rotor and hub
        assembly, transmission, and landing gear. Additionally incorporates
        results of the following correlation scripts:
            fuselage,py
            prop.py
            wing.py
            wiring.py
        Originally written as part of an AA 290 project inteded for trade study
        of the Electric Vectored_Thrust along with the following defined SUAVE vehicle types:
            Electric Multicopter
            Electric Stopped Rotor

        Sources:
        Project Vahana Conceptual Trade Study
        https://github.com/VahanaOpenSource
        
        Inputs:
            config                          SUAVE Config Data Structure
            speed_of_sound                  Local Speed of Sound                [m/s]
            max_tip_mach                    Allowable Tip Mach Number           [Unitless]
            disk_area_factor                Inverse of Disk Area Efficiency     [Unitless]
            max_thrust_to_weight_ratio      Allowable Thrust to Weight Ratio    [Unitless]
            motor_efficiency                Motor Efficiency                    [Unitless]
            
        Outputs:
            output:                         Data Dictionary of Component Masses       [kg]
    """


    
    #-------------------------------------------------------------------------------
    # Unpack Inputs
    #------------------------------------------------------------------------------- 
    rRotor              = config.propulsors.vectored_thrust.rotor.tip_radius 
    bladeSol            = config.propulsors.vectored_thrust.rotor.blade_solidity
    tipMach             = max_tip_mach
    k                   = disk_area_factor
    ToverW              = max_thrust_to_weight_ratio
    etaMotor            = motor_efficiency  

    #-------------------------------------------------------------------------------
    # Assumed Weights
    #-------------------------------------------------------------------------------
    output = Data()
    output.payload      = config.propulsors.vectored_thrust.payload.mass_properties.mass
    output.seats        = 30.
    output.avionics     = 15.
    output.motors       = 10 * config.propulsors.vectored_thrust.number_of_engines
    output.battery      = config.propulsors.vectored_thrust.battery.mass_properties.mass
    output.servos       = 0.65 * config.propulsors.vectored_thrust.number_of_engines 
    output.brs          = 16.
    output.hubs         = 2 * config.propulsors.vectored_thrust.number_of_engines
    output.landing_gear = config.mass_properties.max_takeoff * 0.02

    #-------------------------------------------------------------------------------
    # Calculated Weights
    #-------------------------------------------------------------------------------

    # Preparatory Calculations  
    rho_ref      = 1.225
    Vtip         = speed_of_sound * tipMach                      # Prop Tip Velocity
    omega        = Vtip/rRotor                                   # Prop Ang. Velocity
    maxLift      = config.mass_properties.max_takeoff * ToverW   # Maximum Thrust
    Ct           = maxLift/(rho_ref*np.pi*rRotor**2*Vtip**2)     # Thrust Coefficient 
    AvgCL        = 6 * Ct / bladeSol                             # Average Blade CL
    AvgCD        = 0.012                                         # Average Blade CD  
    maxLiftPower = 1.15*maxLift*(k*np.sqrt(maxLift/(2*rho_ref*np.pi*rRotor**2)) +
                                 bladeSol*AvgCD/8*Vtip**3/(maxLift/(rho_ref*np.pi*rRotor**2)))  
    maxTorque    = maxLiftPower/omega

    # Component Weight Calculations
    num_motors   = 0
    rotor_servos = 0 
    for w in config.wings:
        num_motors += num_motors + len(w.motor_spanwise_locations)
        rotor_servos += 2*len(w.motor_spanwise_locations)
        
    output.rotor_servos     = rotor_servos
    output.lift_rotors      = (prop(config.propulsors.vectored_thrust.rotor, maxLift)* (num_motors)) # make more generic ash jordan about this
    output.fuselage         = fuselage(config)
    output.wiring           = wiring(config, np.ones(8)**0.25, maxLiftPower/etaMotor)

    total_wing_weight = 0.
    for w in config.wings:
        wing_tag = w.tag
        if (wing_tag.find('main_wing') != -1):
            wing_weight = wing(config.wings[w.tag], config, maxLift/5) *Units.kg
            total_wing_weight = total_wing_weight + wing_weight
    output.total_wing_weight = total_wing_weight    


    #-------------------------------------------------------------------------------
    # Weight Summations
    #-------------------------------------------------------------------------------


    output.structural   = (output.lift_rotors +
                           output.hubs +
                            output.fuselage + 
                            output.landing_gear +
                            output.total_wing_weight
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
