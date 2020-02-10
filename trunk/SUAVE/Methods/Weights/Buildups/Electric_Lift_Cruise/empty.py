## @ingroup Methods-Weights-Buildups-Electric_Lift_Cruise

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

## @ingroup Methods-Weights-Buildups-Electric_Lift_Cruise
def empty(config,
          contingency_factor            = 1.1,
          speed_of_sound                = 340.294,
          max_tip_mach                  = 0.65,
          disk_area_factor              = 1.15,
          max_thrust_to_weight_ratio    = 1.1,
          safety_factor                 = 1.5,
          max_g_load                    = 3.8,
          motor_efficiency              = 0.85 * 0.98):
    """weight = SUAVE.Methods.Weights.Buildups.Electric_Lift_Cruise.empty(
            config,
            speed_of_sound              = 340.294,
            max_tip_mach                = 0.65,
            disk_area_factor            = 1.15,
            max_thrust_to_weight_ratio  = 1.1,
            motor_efficiency            = 0.85 * 0.98)
        Calculates the empty fuselage mass for an electric stopped rotor including
        seats, avionics, servomotors, ballistic recovery system, rotor and hub
        assembly, transmission, and landing gear. Additionally incorporates
        results of the following common buildup scripts:
            fuselage,py
            prop.py
            wing.py
            wiring.py
        Originally written as part of an AA 290 project inteded for trade study
        of the Electric Stopped Rotor along with the following defined SUAVE config types:
            Electric Helicopter
            Electric Vectored_Thrust
            
        Sources:
        Project Vahana Conceptual Trade Study
        Inputs:
            config                          SUAVE Config Data Structure
            speed_of_sound                  Local Speed of Sound                [m/s]
            maximumTipMach                  Allowable Tip Mach Number           [Unitless]
            disk_area_factor                Inverse of Disk Area Efficiency     [Unitless]
            max_thrust_to_weight_ratio      Allowable Thrust to Weight Ratio    [Unitless]
            motor_efficiency                Motor Efficiency                    [Unitless]
        
        Outputs:
            output:                         Data Dictionary of Component Masses       [kg]
    """

    output = Data()

#-------------------------------------------------------------------------------
# Unpack Inputs
#-------------------------------------------------------------------------------
    rPropLift           = config.propulsors.propulsor.propeller_lift.tip_radius
    rPropThrust         = config.propulsors.propulsor.propeller_forward.tip_radius
    mBattery            = config.propulsors.propulsor.battery.mass_properties.mass
    mPayload            = config.propulsors.propulsor.payload.mass_properties.mass
    MTOW                = config.mass_properties.max_takeoff
    nLiftProps          = config.propulsors.propulsor.number_of_engines_lift
    nThrustProps        = config.propulsors.propulsor.number_of_engines_forward
    nLiftBlades         = config.propulsors.propulsor.propeller_lift.number_blades
    nThrustBlades       = config.propulsors.propulsor.propeller_forward.number_blades
    fLength             = config.fuselages.fuselage.lengths.total
    fWidth              = config.fuselages.fuselage.width
    fHeight             = config.fuselages.fuselage.heights.maximum
    maxSpan             = config.wings['main_wing'].spans.projected
    
    sound               = speed_of_sound
    tipMach             = max_tip_mach
    k                   = disk_area_factor
    ToverW              = max_thrust_to_weight_ratio
    etaMotor            = motor_efficiency

    output.payload          = mPayload * Units.kg
    output.seats            = 30. *Units.kg
    output.avionics         = 15. *Units.kg
    output.motors           = (config.propulsors.propulsor.number_of_engines_lift * 10. *Units.kg 
                               + config.propulsors.propulsor.number_of_engines_forward * 25. *Units.kg)
    output.battery          = mBattery *Units.kg
    output.servos           = config.propulsors.propulsor.number_of_engines_lift * 0.65 *Units.kg
    output.brs              = 16. *Units.kg
    output.hubs             = (config.propulsors.propulsor.number_of_engines_lift * 2. *Units.kg
                               + config.propulsors.propulsor.number_of_engines_forward * 5. *Units.kg)
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

    output.lift_rotors      = (prop(config.propulsors.propulsor.propeller_lift, maxLift) 
                               * (len(config.wings['main_wing'].motor_spanwise_locations))) *Units.kg
    output.thrust_rotors    = prop(config.propulsors.propulsor.propeller_forward, maxLift/5) *Units.kg
    output.fuselage         = fuselage(config) *Units.kg
    output.wiring           = wiring(config,
                                     np.ones(8)**0.25,
                                     maxLiftPower/etaMotor) *Units.kg

    total_wing_weight = 0.
    output.wings = Data()


    
    
    for w in config.wings:
        wing_tag = w.tag 
        if (wing_tag.find('main_wing') != -1):
            wing_weight = wing(config.wings[w.tag],
                               config, 
                               maxLift/5, safety_factor= safety_factor, max_g_load =  max_g_load ) *Units.kg
            tag = wing_tag
            output.wings[tag] = wing_weight
            total_wing_weight = total_wing_weight + wing_weight
    output.total_wing_weight = total_wing_weight
    
#-------------------------------------------------------------------------------
# Weight Summations
#-------------------------------------------------------------------------------
    output.structural   = (output.lift_rotors +
                            output.thrust_rotors +
                            output.hubs +
                            output.fuselage + 
                            output.landing_gear +
                            output.total_wing_weight
                            ) *Units.kg

    output.empty        = (contingency_factor * (
                            output.structural +
                            output.seats +
                            output.avionics +
                            output.motors +
                            output.servos +
                            output.wiring +
                            output.brs
                            ) + output.battery) *Units.kg
    
    output.total        = (output.empty +
                            output.payload) *Units.kg

    return output