## @ingroup Methods-Weights-Buildups-EVTOL
# empty.py
#
# Created: Apr, 2019, J. Smart
# Modified:

#-------------------------------------------------------------------------------
# Imports
#-------------------------------------------------------------------------------

from SUAVE.Core import Units, Data

from SUAVE.Methods.Weights.Buildups.Common.fuselage import fuselage
from SUAVE.Methods.Weights.Buildups.Common.prop import prop
from SUAVE.Methods.Weights.Buildups.Common.wiring import wiring
from SUAVE.Methods.Weights.Buildups.Common.wing import wing

from SUAVE.Components.Energy.Networks import Battery_Propeller
from SUAVE.Components.Energy.Networks import Lift_Forward_Propulsor

import numpy as np
from warnings import  warn


#-------------------------------------------------------------------------------
# Empty
#-------------------------------------------------------------------------------

## @ingroup Methods-Weights-Buildups-EVTOL

def empty(config,
          speed_of_sound                = 340.294,
          max_tip_mach                  = 0.65,
          disk_area_factor              = 1.15,
          max_thrust_to_weight_ratio    = 1.1,
          motor_efficiency              = 0.85 * 0.98):

    """mass = SUAVE.Methods.Weights.Buildups.EVTOL.empty(
            config,
            speed_of_sound                = 340.294,
            max_tip_mach                  = 0.65,
            disk_area_factor              = 1.15,
            max_thrust_to_weight_ratio    = 1.1,
            motor_efficiency              = 0.85 * 0.98)

        Calculates the empty vehicle mass for an EVTOL-type aircraft including seats,
        avionics, servomotors, ballistic recovery system, rotor and hub assembly,
        transmission, and landing gear. Incorporates the results of the following
        common-use buildups:

            fuselage.py
            prop.py
            wing.py
            wiring.py

        Inputs:

            config:                     SUAVE Config Data Stucture
            speed_of_sound:             Local Speed of Sound                [m/s]
            max_tip_mach:               Allowable Tip Mach Number           [Unitless]
            disk_area_factor:           Inverse of Disk Area Efficiency     [Unitless]
            max_thrust_to_weight_ratio: Allowable Thrust to Weight Ratio    [Unitless]
            motor_efficiency:           Motor Efficiency                    [Unitless]

        Outpus:

            outputs:                    Data Dictionary of Component Masses [kg]

        Output data dictionary has the following book-keeping hierarchical structure:

            Output
                Total
                    Empty
                        Structural
                            Fuselage
                            Wings
                            Landing Gear
                            Rotors
                            Hubs
                        Seats
                        Battery
                        Motors
                        Servomotors
                        Wiring
                    Systems
                        Avionics
                        ECS
                        BRS
                    Payload


    """

    output = Data()

#-------------------------------------------------------------------------------
# Unpacking Inputs
#-------------------------------------------------------------------------------

    propulsor = config.propulsors.propulsor

    # Common Inputs

    mBattery    = propulsor.battery.mass_properties.mass
    mPayload    = propulsor.payload.mass_properties.mass
    MTOW        = config.mass_properties.max_takeoff
    fLength     = config.fuselages.fuselage.lengths.total
    fWidth      = config.fuselages.fuselage.width
    fHeight     = config.fuselages.fuselage.heights.maximum

    # Conditional Inputs

    if isinstance(propulsor, Battery_Propeller):
        nLiftProps      = propulsor.number_of_engines
        nLiftBlades     = propulsor.propeller.prop_attributes.number_blades
        rTipLiftProp    = propulsor.propeller.prop_attributes.tip_radius
        rHubLiftProp    = propulsor.propeller.prop_attributes.hub_radius
        cLiftProp       = propulsor.propeller.prop_attributes.chord_distribution

    elif isinstance(propulsor, Lift_Forward_Propulsor):
        nLiftProps      = propulsor.number_of_engines_lift
        nLiftBlades     = propulsor.propeller_lift.prop_attributes.number_blades
        rTipLiftProp    = propulsor.propeller_lift.prop_attributes.tip_radius
        rHubLiftProp    = propulsor.propeller_lift.prop_attributes.hub_radius
        cLiftProp       = propulsor.propeller_lift.prop_attributes.chord_distribution

        nThrustProps    = propulsor.number_of_engines_forward
        nThrustBlades   = propulsor.propeller_forward.prop_attributes.number_blades
        rTipThrustProp  = propulsor.propeller_forward.prop_attributes.tip_radius
        rHubThrustProp  = propulsor.propeller_forward.prop_attributes.hub_radius
    else:
        warn("""eVTOL weight buildup only supports the Battery Propeller and Lift Forward Propulsor energy networks.\n
        Weight buildup will not return information on propulsion system.""", stacklevel=1)

    sound       = speed_of_sound
    tipMach     = max_tip_mach
    k           = disk_area_factor
    ToverW      = max_thrust_to_weight_ratio
    etaMotor    = motor_efficiency

#-------------------------------------------------------------------------------
# Fixed Weights
#-------------------------------------------------------------------------------

    output.payload      = mPayload                  * Units.kg
    output.seats        = config.passengers * 15.   * Units.kg
    output.avionics     = 15.                       * Units.kg
    output.battery      = mBattery                  * Units.kg
    output.landing_gear = MTOW * 0.02               * Units.kg

    if isinstance(propulsor, Battery_Propeller):
        output.servos   = 5.2 * nLiftProps          * Units.kg
        output.hubs     = MTOW * 0.04 * nLiftProps  * Units.kg
        if nLiftProps > 1:
            output.BRS = 16.                        * Units.kg

    elif isinstance(propulsor, Lift_Forward_Propulsor):
        output.servos   = 0.65  * (nLiftProps + nThrustProps)   * Units.kg
        output.hubs     = 2.    * (nLiftProps + nThrustProps)   * Units.kg
        output.BRS      = 16.                                   * Units.kg

#-------------------------------------------------------------------------------
# Calculated Attributes
#-------------------------------------------------------------------------------

    # Preparatory Calculations:

    maxVTip         = sound * tipMach                               # Maximum Tip Velocity
    maxLift         = MTOW * ToverW                                 # Maximum Lift
    liftMeanRad     = ((rTipLiftProp**2 + rHubLiftProp**2)/2)**0.5  # Propeller Mean Radius
    liftPitch       = 2*np.pi*liftMeanRad/nLiftBlades               # Propeller Pitch
    liftBladeSol    = cLiftProp/liftPitch                           # Blade Solidity
    AvgLiftBladeCD  = 0.012                                         # Assumed Drag Coeff.
    psuedoCT        = maxLift/(1.225*np.pi*0.8**2)

    maxLiftPower    = 1.15 * maxLift * (
            k * np.sqrt(psuedoCT/2.) +
            liftBladeSol * AvgLiftBladeCD/8. * maxVTip**3/psuedoCT
    )



#-------------------------------------------------------------------------------
# Component Weight Calculations
#-------------------------------------------------------------------------------

#-------------------------------------------------------------------------------
# Pack Up Outputs
#-------------------------------------------------------------------------------