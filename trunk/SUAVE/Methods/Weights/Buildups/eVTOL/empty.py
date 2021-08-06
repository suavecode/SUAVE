## @ingroup Methods-Weights-Buildups-eVTOL
# empty.py
#
# Created:    Apr, 2019, J. Smart
# Modified:   July, 2021, R. Erhard

#-------------------------------------------------------------------------------
# Imports
#-------------------------------------------------------------------------------
import SUAVE
from SUAVE.Core import Units, Data

from SUAVE.Methods.Weights.Buildups.Common.fuselage import fuselage
from SUAVE.Methods.Weights.Buildups.Common.prop import prop
from SUAVE.Methods.Weights.Buildups.Common.wiring import wiring
from SUAVE.Methods.Weights.Buildups.Common.wing import wing

from SUAVE.Components.Energy.Networks import Battery_Propeller
from SUAVE.Components.Energy.Networks import Lift_Cruise

import numpy as np
from warnings import  warn


#-------------------------------------------------------------------------------
# Empty
#-------------------------------------------------------------------------------

## @ingroup Methods-Weights-Buildups-eVTOL

def empty(config,
          contingency_factor            = 1.1,
          speed_of_sound                = 340.294,
          max_tip_mach                  = 0.65,
          disk_area_factor              = 1.15,
          safety_factor                 = 1.5,
          max_thrust_to_weight_ratio    = 1.1,
          max_g_load                    = 3.8,
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

        Sources:
        Project Vahana Conceptual Trade Study
        https://github.com/VahanaOpenSource


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
                Total.
                    Empty.
                        Structural.
                            Fuselage
                            Wings
                            Landing Gear
                            Rotors
                            Hubs
                        Seats
                        Battery
                        Motors
                        Servo
                    Systems.
                        Avionics
                        ECS               - Environmental Control System
                        BRS               - Ballistic Recovery System
                        Wiring            - Aircraft Electronic Wiring
                    Payload

    """

    # Set up data structures for SUAVE weight methods
    output                   = Data()
    output.lift_rotors       = 0.0
    output.propellers        = 0.0
    output.lift_rotor_motors = 0.0
    output.propeller_motors  = 0.0
    output.battery           = 0.0
    output.payload           = 0.0
    output.servos            = 0.0
    output.hubs              = 0.0
    output.BRS               = 0.0

    config.payload.passengers                      = SUAVE.Components.Physical_Component()
    config.payload.baggage                         = SUAVE.Components.Physical_Component()
    config.payload.cargo                           = SUAVE.Components.Physical_Component()
    control_systems                                = SUAVE.Components.Physical_Component()
    electrical_systems                             = SUAVE.Components.Physical_Component()
    furnishings                                    = SUAVE.Components.Physical_Component()
    air_conditioner                                = SUAVE.Components.Physical_Component()
    fuel                                           = SUAVE.Components.Physical_Component()
    apu                                            = SUAVE.Components.Physical_Component()
    hydraulics                                     = SUAVE.Components.Physical_Component()
    avionics                                       = SUAVE.Components.Energy.Peripherals.Avionics()
    optionals                                      = SUAVE.Components.Physical_Component()

    # assign components to vehicle
    config.systems.control_systems                 = control_systems
    config.systems.electrical_systems              = electrical_systems
    config.systems.avionics                        = avionics
    config.systems.furnishings                     = furnishings
    config.systems.air_conditioner                 = air_conditioner
    config.systems.fuel                            = fuel
    config.systems.apu                             = apu
    config.systems.hydraulics                      = hydraulics
    config.systems.optionals                       = optionals


    #-------------------------------------------------------------------------------
    # Fixed Weights
    #-------------------------------------------------------------------------------
    MTOW                = config.mass_properties.max_takeoff
    output.seats        = config.passengers * 15.   * Units.kg
    output.passengers   = config.passengers * 70.   * Units.kg
    output.avionics     = 15.                       * Units.kg
    output.landing_gear = MTOW * 0.02               * Units.kg
    output.ECS          = config.passengers * 7.    * Units.kg

    # Inputs and other constants
    tipMach        = max_tip_mach
    k              = disk_area_factor
    ToverW         = max_thrust_to_weight_ratio
    eta            = motor_efficiency
    rho_ref        = 1.225
    maxVTip        = speed_of_sound * tipMach         # Prop Tip Velocity
    maxLift        = MTOW * ToverW * 9.81             # Maximum Thrust
    AvgBladeCD     = 0.012                            # Average Blade CD

    # Select a length scale depending on what kind of vehicle this is
    length_scale = 1.
    nose_length  = 0.

    # Check if there is a fuselage
    C =  SUAVE.Components
    if len(config.fuselages) == 0.:
        for w  in config.wings:
            if isinstance(w ,C.Wings.Main_Wing):
                b = wing.chords.root
                if b>length_scale:
                    length_scale = b
                    nose_length  = 0.25*b
    else:
        for fuse in config.fuselages:
            nose   = fuse.lengths.nose
            length = fuse.lengths.total
            if length > length_scale:
                length_scale = length
                nose_length  = nose

    #-------------------------------------------------------------------------------
    # Environmental Control System
    #-------------------------------------------------------------------------------
    config.systems.air_conditioner.origin[0][0]          = 0.51 * length_scale
    config.systems.air_conditioner.mass_properties.mass  = output.ECS

    #-------------------------------------------------------------------------------
    # Network Weight
    #-------------------------------------------------------------------------------
    for network in config.networks:

        #-------------------------------------------------------------------------------
        # Battery Weight
        #-------------------------------------------------------------------------------
        network.battery.origin[0][0]                                   = 0.51 * length_scale
        network.battery.mass_properties.center_of_gravity[0][0]        = 0.0
        output.battery                                                += network.battery.mass_properties.mass * Units.kg

        #-------------------------------------------------------------------------------
        # Payload Weight
        #-------------------------------------------------------------------------------
        network.payload.origin[0][0]                                   = 0.51 * length_scale
        network.payload.mass_properties.center_of_gravity[0][0]        = 0.0
        output.payload                                                += network.payload.mass_properties.mass * Units.kg

        #-------------------------------------------------------------------------------
        # Avionics Weight
        #-------------------------------------------------------------------------------
        network.avionics.origin[0][0]                                  = 0.4 * nose_length
        network.avionics.mass_properties.center_of_gravity[0][0]       = 0.0
        network.avionics.mass_properties.mass                          = output.avionics


        #-------------------------------------------------------------------------------
        # Servo, Hub and BRS Weights
        #-------------------------------------------------------------------------------

        lift_rotor_hub_weight   = 4.   * Units.kg
        prop_hub_weight         = MTOW * 0.04  * Units.kg

        lift_rotor_BRS_weight   = 16.  * Units.kg



        #-------------------------------------------------------------------------------
        # Rotor, Propeller, parameters for sizing
        #-------------------------------------------------------------------------------
        if isinstance(network, Lift_Cruise):
            # Total number of rotors and propellers
            nLiftRotors   = network.number_of_lift_rotor_engines
            nThrustProps  = network.number_of_propeller_engines
            props         = network.propellers
            rots          = network.lift_rotors
            prop_motors   = network.propeller_motors
            rot_motors    = network.lift_rotor_motors


        elif isinstance(network, Battery_Propeller):
            # Total number of rotors and propellers
            nLiftRotors   = 0
            nThrustProps  = network.number_of_propeller_engines
            props         = network.propellers
            prop_motors   = network.propeller_motors

        else:
            warn("""eVTOL weight buildup only supports the Battery Propeller and Lift Cruise energy networks.\n
            Weight buildup will not return information on propulsion system.""", stacklevel=1)

        nProps  = int(nLiftRotors + nThrustProps)
        if nProps > 1:
            prop_BRS_weight     = 16.   * Units.kg
        else:
            prop_BRS_weight     = 0.   * Units.kg

        prop_servo_weight  = 0.0

        if nThrustProps > 0:
            if network.identical_propellers:
                # Get reference properties for sizing from first propeller (assumes identical)
                proprotor    = props[list(props.keys())[0]]
                propmotor    = prop_motors[list(prop_motors.keys())[0]]
                rTip_ref     = proprotor.tip_radius
                bladeSol_ref = proprotor.blade_solidity

                if proprotor.variable_pitch:
                    prop_servo_weight  = 5.2  * Units.kg

                # Compute and add propeller weights
                propeller_mass                 = prop(proprotor, maxLift/5.) * Units.kg
                output.propellers             += nThrustProps * propeller_mass
                output.propeller_motors       += nThrustProps * propmotor.mass_properties.mass
                proprotor.mass_properties.mass = propeller_mass + prop_hub_weight + prop_servo_weight

            else:
                for idx, propeller in enumerate(network.propellers):
                    proprotor    = propeller
                    propmotor    = prop_motors[list(prop_motors.keys())[idx]]
                    rTip_ref     = proprotor.tip_radius
                    bladeSol_ref = proprotor.blade_solidity

                    if proprotor.variable_pitch:
                        prop_servo_weight  = 5.2  * Units.kg

                    # Compute and add propeller weights
                    propeller_mass                 = prop(proprotor, maxLift/5.) * Units.kg
                    output.propellers             += propeller_mass
                    output.propeller_motors       += propmotor.mass_properties.mass
                    proprotor.mass_properties.mass = propeller_mass + prop_hub_weight + prop_servo_weight

        lift_rotor_servo_weight = 0.0
        if nLiftRotors > 0:
            if network.identical_lift_rotors:
                # Get reference properties for sizing from first lift_rotor (assumes identical)
                liftrotor = rots[list(rots.keys())[0]]
                liftmotor = rot_motors[list(rot_motors.keys())[0]]
                rTip_ref     = liftrotor.tip_radius
                bladeSol_ref = liftrotor.blade_solidity


                if liftrotor.variable_pitch:
                    lift_rotor_servo_weight = 0.65 * Units.kg

                # Compute and add lift_rotor weights
                lift_rotor_mass                = prop(liftrotor, maxLift / max(nLiftRotors - 1, 1))  * Units.kg
                output.lift_rotors            += nLiftRotors * lift_rotor_mass
                output.lift_rotor_motors      += nLiftRotors * liftmotor.mass_properties.mass
                liftrotor.mass_properties.mass = lift_rotor_mass + lift_rotor_hub_weight + lift_rotor_servo_weight

            else:
                for idx, lift_rotor in enumerate(network.lift_rotors):
                    liftrotor    = lift_rotor
                    liftmotor    = rot_motors[list(rot_motors.keys())[idx]]
                    rTip_ref     = liftrotor.tip_radius
                    bladeSol_ref = liftrotor.blade_solidity


                    if liftrotor.variable_pitch:
                        lift_rotor_servo_weight = 0.65 * Units.kg

                    # Compute and add lift_rotor weights
                    lift_rotor_mass                = prop(liftrotor, maxLift / max(nLiftRotors - 1, 1))  * Units.kg
                    output.lift_rotors            += lift_rotor_mass
                    output.lift_rotor_motors      += liftmotor.mass_properties.mass
                    liftrotor.mass_properties.mass = lift_rotor_mass + lift_rotor_hub_weight + lift_rotor_servo_weight

        # Add associated weights
        output.servos += (nLiftRotors * lift_rotor_servo_weight + nThrustProps * prop_servo_weight)
        output.hubs   += (nLiftRotors * lift_rotor_hub_weight + nThrustProps * prop_hub_weight)
        output.BRS    += (prop_BRS_weight + lift_rotor_BRS_weight)

        maxLiftPower   = 1.15*maxLift*(k*np.sqrt(maxLift/(2*rho_ref*np.pi*rTip_ref**2)) +
                                           bladeSol_ref*AvgBladeCD/8*maxVTip**3/(maxLift/(rho_ref*np.pi*rTip_ref**2)))
        # Tail Rotor
        if nLiftRotors == 1: # this assumes that the vehicle is an electric helicopter with a tail rotor
            
            maxLiftOmega   = maxVTip/rTip_ref
            maxLiftTorque  = maxLiftPower / maxLiftOmega

            tailrotor = next(iter(network.lift_rotors))
            output.tail_rotor   = prop(tailrotor, 1.5*maxLiftTorque/(1.25*rTip_ref))*0.2 * Units.kg
            output.lift_rotors += output.tail_rotor

    # sum motor weight
    output.motors = output.lift_rotor_motors + output.propeller_motors

    #-------------------------------------------------------------------------------
    # Wing and Motor Wiring Weight
    #-------------------------------------------------------------------------------
    total_wing_weight   = 0.0
    total_wiring_weight = 0.0
    output.wings        = Data()
    output.wiring       = Data()

    for w in config.wings:
        if w.symbolic:
            wing_weight = 0
        else:
            wing_weight            = wing(w, config, maxLift/5, safety_factor= safety_factor, max_g_load =  max_g_load )
            wing_tag               = w.tag
            output.wings[wing_tag] = wing_weight
            w.mass_properties.mass = wing_weight

        total_wing_weight    = total_wing_weight + wing_weight

        # wiring weight
        wiring_weight        = wiring(w, config, maxLiftPower/(eta*nProps)) * Units.kg
        total_wiring_weight  = total_wiring_weight + wiring_weight

    output.wiring            = total_wiring_weight
    output.total_wing_weight = total_wing_weight

    #-------------------------------------------------------------------------------
    # Landing Gear Weight
    #-------------------------------------------------------------------------------
    if not hasattr(config.landing_gear, 'nose'):
        config.landing_gear.nose       = SUAVE.Components.Landing_Gear.Nose_Landing_Gear()
    config.landing_gear.nose.mass      = 0.0
    if not hasattr(config.landing_gear, 'main'):
        config.landing_gear.main       = SUAVE.Components.Landing_Gear.Main_Landing_Gear()
    config.landing_gear.main.mass      = output.landing_gear

    #-------------------------------------------------------------------------------
    # Fuselage  Weight
    #-------------------------------------------------------------------------------
    output.fuselage = fuselage(config) * Units.kg
    config.fuselages.fuselage.mass_properties.center_of_gravity[0][0] = .45*config.fuselages.fuselage.lengths.total
    config.fuselages.fuselage.mass_properties.mass                    =  output.fuselage + output.passengers + output.seats +\
                                                                         output.wiring + output.BRS

    #-------------------------------------------------------------------------------
    # Pack Up Outputs
    #-------------------------------------------------------------------------------
    output.structural = (output.lift_rotors + output.propellers + output.hubs +
                                 output.fuselage + output.landing_gear +output.total_wing_weight)*Units.kg

    output.empty      = (contingency_factor * (output.structural + output.seats + output.avionics +output.ECS +\
                        output.motors + output.servos + output.wiring + output.BRS) + output.battery) *Units.kg

    output.total      = output.empty + output.payload + output.passengers

    return output
