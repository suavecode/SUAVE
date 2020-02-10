## @ingroup Methods-Weights-Buildups-Electric_Vectored_Thrust
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
from SUAVE.Methods.Weights.Buildups.Common.wing import wing
import numpy as np

#-------------------------------------------------------------------------------
# Empty
#-------------------------------------------------------------------------------

## @ingroup Methods-Weights-Buildups-Electric_Vectored_Thrust
def empty(config,
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
            motor_efficience            = 0.85 * 0.98)
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
            Electric Helicopter
            Electric Stopped Rotor

        Sources:
        Project Vahana Conceptual Trade Study
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

    output = Data()

#-------------------------------------------------------------------------------
# Assumed Weights
#-------------------------------------------------------------------------------

    output.payload          = config.propulsors.propulsor.payload.mass_properties.mass
    output.seats            = 30.
    output.avionics         = 15.
    output.motors           = 10 * config.propulsors.propulsor.number_of_engines
    output.battery          = config.propulsors.propulsor.battery.mass_properties.mass
    output.servos           = 0.65 * config.propulsors.propulsor.number_of_engines
    output.rotor_servos     = 2 * (len(config.wings['main_wing'].motor_spanwise_locations) 
                                   + len(config.wings['main_wing_2'].motor_spanwise_locations))
    output.brs              = 16.
    output.hubs             = 2 * config.propulsors.propulsor.number_of_engines
    output.landing_gear     = config.mass_properties.max_takeoff * 0.02

#-------------------------------------------------------------------------------
# Calculated Weights
#-------------------------------------------------------------------------------

    # Preparatory Calculations


    sound               = speed_of_sound
    tipMach             = max_tip_mach
    k                   = disk_area_factor
    ToverW              = max_thrust_to_weight_ratio
    etaMotor            = motor_efficiency 
    Vtip                = sound * tipMach                               # Prop Tip Velocity
    omega               = Vtip/0.8                                      # Prop Ang. Velocity
    maxLift             = config.mass_properties.max_takeoff * ToverW   # Maximum Thrust
    Ct                  = maxLift/(1.225*np.pi*0.8**2*Vtip**2)          # Thrust Coefficient
    bladeSol            = 0.1                                           # Blade Solidity
    AvgCL               = 6 * Ct / bladeSol                             # Average Blade CL
    AvgCD               = 0.012                                         # Average Blade CD 
    
    maxLiftPower        = 1.15*maxLift*(
        k*np.sqrt(maxLift/(2*1.225*np.pi*0.8**2)) +
                    bladeSol*AvgCD/8*Vtip**3/(maxLift/(1.225*np.pi*0.8**2))
    )

    maxTorque = maxLiftPower/omega

    # Component Weight Calculations
    num_motors = 0
    for w in config.wings:
        num_motors += num_motors + len(w.motor_spanwise_locations)
        
    output.lift_rotors      = (prop(config.propulsors.propulsor.propeller, maxLift)* (num_motors)) # make more generic ash jordan about this
    output.fuselage         = fuselage(config)
    output.wiring           = wiring(config,
                                     np.ones(8)**0.25,
                                     maxLiftPower/etaMotor)

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
